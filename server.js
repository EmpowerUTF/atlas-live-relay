import http from "node:http";
import crypto from "node:crypto";
import { URL } from "node:url";
import { GoogleGenAI, Modality } from "@google/genai";
import { WebSocket, WebSocketServer } from "ws";

const PORT = Number(process.env.PORT || 3000);
const RELAY_VERSION = "3.2.0-persistent-device-session";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const ATLAS_DEVICE_TOKEN = process.env.ATLAS_DEVICE_TOKEN || "";
const GEMINI_PRIMARY_MODEL =
  process.env.GEMINI_MODEL || "gemini-3.1-flash-live-preview";
const GEMINI_FALLBACK_MODEL =
  process.env.GEMINI_FALLBACK_MODEL ||
  "gemini-2.5-flash-native-audio-preview-12-2025";
const GEMINI_RETRY_DELAY_MS = 1500;
const ATLAS_VOICE = "Orus";
const BASE_SYSTEM_INSTRUCTION =
  process.env.ATLAS_SYSTEM_INSTRUCTION ||
  "You are Atlas, a fast, practical AI companion. Speak naturally and directly. " +
    "Keep routine replies concise. Ask one useful follow-up only when it is genuinely needed.";

const LANGUAGE_AND_VOICE_LOCK =
  "LANGUAGE AND VOICE REQUIREMENT: Always speak in English only. " +
  "Never answer in Spanish or any other language, even if automatic transcription " +
  "mistakenly labels the owner's English speech as another language. " +
  "Use a natural adult male voice with a clear, neutral American English accent. " +
  "Do not imitate the language or accent inferred from noisy audio. " +
  "Only change language if the owner explicitly says the exact words: switch language.";

const SUPABASE_URL = (process.env.SUPABASE_URL || "").replace(/\/$/, "");
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || "";
const SUPABASE_ENABLED = Boolean(SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY);

const MAX_ATLAS_MESSAGE_BYTES = 64 * 1024;
const PCM_SLICE_BYTES = 960; // 20 ms of 24 kHz mono PCM16.
const PCM_SLICE_MS = 20;
const ATLAS_BACKPRESSURE_BYTES = 96 * 1024;
const SESSION_HANDLE_MAX_AGE_MS = 90 * 60 * 1000;
const RECENT_HISTORY_TURNS = 10;
const TRANSCRIPT_SETTLE_MS = 450;
const ATLAS_HEARTBEAT_INTERVAL_MS = 20_000;
const ATLAS_STALE_TIMEOUT_MS = 120_000;
const ATLAS_REATTACH_GRACE_MS = 180_000;

if (!GEMINI_API_KEY) {
  console.error("Missing required environment variable: GEMINI_API_KEY");
  process.exit(1);
}
if (!ATLAS_DEVICE_TOKEN) {
  console.error("Missing required environment variable: ATLAS_DEVICE_TOKEN");
  process.exit(1);
}

const googleAi = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

function jsonResponse(res, status, payload) {
  if (res.writableEnded) return;
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
    "cache-control": "no-store",
  });
  res.end(body);
}

function authorized(parsed) {
  return parsed.searchParams.get("token") === ATLAS_DEVICE_TOKEN;
}

function readJsonBody(req, maxBytes = 32 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let total = 0;
    req.on("data", (chunk) => {
      total += chunk.length;
      if (total > maxBytes) {
        reject(Object.assign(new Error("body_too_large"), { status: 413 }));
        req.destroy();
        return;
      }
      chunks.push(Buffer.from(chunk));
    });
    req.on("end", () => {
      try {
        const text = Buffer.concat(chunks, total).toString("utf8");
        resolve(text ? JSON.parse(text) : {});
      } catch {
        reject(Object.assign(new Error("invalid_json"), { status: 400 }));
      }
    });
    req.on("error", reject);
  });
}

function safeDeviceId(value) {
  const cleaned = String(value || "atlas-v1").replace(/[^a-zA-Z0-9_.-]/g, "");
  return cleaned.slice(0, 64) || "atlas-v1";
}

function mergeTranscript(current, fragment) {
  const next = String(fragment || "").trim();
  if (!next) return current;
  const existing = String(current || "").trim();
  if (!existing) return next;
  if (existing === next || existing.endsWith(next)) return existing;
  if (next.startsWith(existing)) return next;
  return `${existing}${/^[,.;:!?]/.test(next) ? "" : " "}${next}`;
}

async function supabaseRequest(path, { method = "GET", body, prefer } = {}) {
  if (!SUPABASE_ENABLED) return null;
  const headers = {
    apikey: SUPABASE_SERVICE_ROLE_KEY,
    authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
    "content-type": "application/json",
  };
  if (prefer) headers.prefer = prefer;

  const response = await fetch(`${SUPABASE_URL}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  const text = await response.text();
  if (!response.ok) {
    throw new Error(`Supabase ${response.status}: ${text.slice(0, 500)}`);
  }
  return text ? JSON.parse(text) : null;
}

async function loadSessionState(deviceId) {
  if (!SUPABASE_ENABLED) return null;
  const rows = await supabaseRequest(
    `/rest/v1/atlas_session_state?device_id=eq.${encodeURIComponent(deviceId)}` +
      `&select=device_id,conversation_id,resume_handle,updated_at&limit=1`,
  );
  return rows?.[0] || null;
}

async function upsertSessionState(deviceId, conversationId, resumeHandle) {
  if (!SUPABASE_ENABLED || !resumeHandle) return;
  await supabaseRequest("/rest/v1/atlas_session_state?on_conflict=device_id", {
    method: "POST",
    prefer: "resolution=merge-duplicates,return=minimal",
    body: {
      device_id: deviceId,
      conversation_id: conversationId,
      resume_handle: resumeHandle,
      updated_at: new Date().toISOString(),
    },
  });
}

async function clearSessionState(deviceId) {
  if (!SUPABASE_ENABLED) return;
  await supabaseRequest(
    `/rest/v1/atlas_session_state?device_id=eq.${encodeURIComponent(deviceId)}`,
    { method: "DELETE", prefer: "return=minimal" },
  );
}

async function createConversation(deviceId) {
  const id = crypto.randomUUID();
  if (SUPABASE_ENABLED) {
    await supabaseRequest("/rest/v1/atlas_conversations", {
      method: "POST",
      prefer: "return=minimal",
      body: {
        id,
        device_id: deviceId,
        started_at: new Date().toISOString(),
        last_active_at: new Date().toISOString(),
      },
    });
  }
  return id;
}

async function touchConversation(conversationId) {
  if (!SUPABASE_ENABLED || !conversationId) return;
  await supabaseRequest(
    `/rest/v1/atlas_conversations?id=eq.${encodeURIComponent(conversationId)}`,
    {
      method: "PATCH",
      prefer: "return=minimal",
      body: { last_active_at: new Date().toISOString() },
    },
  );
}

async function loadRecentTurns(deviceId) {
  if (!SUPABASE_ENABLED) return [];
  const rows = await supabaseRequest(
    `/rest/v1/atlas_turns?device_id=eq.${encodeURIComponent(deviceId)}` +
      `&select=user_text,atlas_text,created_at&order=created_at.desc&limit=${RECENT_HISTORY_TURNS}`,
  );
  return Array.isArray(rows) ? rows.reverse() : [];
}

async function loadMemories(deviceId) {
  if (!SUPABASE_ENABLED) return [];
  const rows = await supabaseRequest(
    `/rest/v1/atlas_memories?device_id=eq.${encodeURIComponent(deviceId)}` +
      "&active=eq.true&select=memory_key,content,importance&order=importance.desc,updated_at.desc&limit=30",
  );
  return Array.isArray(rows) ? rows : [];
}

async function saveTurn(turn) {
  if (!SUPABASE_ENABLED) return;
  await supabaseRequest("/rest/v1/atlas_turns", {
    method: "POST",
    prefer: "return=minimal",
    body: {
      conversation_id: turn.conversationId,
      device_id: turn.deviceId,
      turn_index: turn.turnIndex,
      user_text: turn.inputTranscript || null,
      atlas_text: turn.outputTranscript || null,
      input_bytes: turn.inputBytes,
      output_bytes: turn.outputBytes,
      first_audio_latency_ms: turn.firstAudioLatencyMs || null,
      total_turn_ms: turn.totalTurnMs || null,
      usage: turn.usage || {},
      created_at: new Date(turn.startedAt).toISOString(),
    },
  });
  await touchConversation(turn.conversationId);
}

function buildSystemInstruction(memories, recentTurns = []) {
  let instruction = `${BASE_SYSTEM_INSTRUCTION}\n\n${LANGUAGE_AND_VOICE_LOCK}`;

  if (memories.length) {
    const memoryLines = memories.map((item) => `- ${item.content}`).join("\n");
    instruction +=
      "\n\nDurable owner memory supplied by Atlas storage. " +
      "Use it only when relevant; do not mention the storage system:\n" +
      memoryLines;
  }

  if (recentTurns.length) {
    const recentLines = [];
    for (const turn of recentTurns.slice(-6)) {
      if (turn.user_text) recentLines.push(`Owner: ${turn.user_text}`);
      if (turn.atlas_text) recentLines.push(`Atlas: ${turn.atlas_text}`);
    }
    if (recentLines.length) {
      instruction +=
        "\n\nRecent saved conversation context. Continue naturally when relevant:\n" +
        recentLines.join("\n");
    }
  }

  return instruction;
}

function historyTurns(recentTurns) {
  const turns = [];
  for (const turn of recentTurns) {
    if (turn.user_text) {
      turns.push({ role: "user", parts: [{ text: turn.user_text }] });
    }
    if (turn.atlas_text) {
      turns.push({ role: "model", parts: [{ text: turn.atlas_text }] });
    }
  }
  return turns;
}



class PcmPacer {
  constructor(getAtlasSocket, onDrained) {
    this.getAtlasSocket = getAtlasSocket;
    this.onDrained = onDrained;
    this.buffer = Buffer.alloc(0);
    this.timer = null;
    this.done = false;
    this.closed = false;
    this.sending = false;
    this.started = false;
    this.nextSendAt = 0;
  }

  enqueue(pcm) {
    if (this.closed || !pcm?.length) return;
    this.buffer = this.buffer.length
      ? Buffer.concat([this.buffer, pcm])
      : Buffer.from(pcm);
    this.wake();
  }

  markDone() {
    if (this.closed) return;
    this.done = true;
    this.wake();
  }

  clear() {
    this.closed = true;
    this.buffer = Buffer.alloc(0);
    this.sending = false;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  wake() {
    if (this.closed || this.sending || this.timer) return;
    this.schedule(0);
  }

  schedule(delayMs) {
    if (this.closed || this.timer) return;
    this.timer = setTimeout(() => this.tick(), Math.max(0, delayMs));
  }

  tick() {
    this.timer = null;
    if (this.closed || this.sending) return;

    const atlas = this.getAtlasSocket();
    if (!atlas || atlas.readyState !== WebSocket.OPEN) {
      // Keep queued PCM during a brief ESP32 reconnect. The DeviceBridge owns
      // the Gemini session independently and calls wake() after reattachment.
      this.schedule(50);
      return;
    }

    if (atlas.bufferedAmount > ATLAS_BACKPRESSURE_BYTES) {
      this.schedule(5);
      return;
    }

    const available = this.buffer.length;
    const sendLength = available >= PCM_SLICE_BYTES
      ? PCM_SLICE_BYTES
      : (this.done ? (available & ~1) : 0);

    if (sendLength > 0) {
      const slice = Buffer.from(this.buffer.subarray(0, sendLength));

      if (!this.started) {
        this.started = true;
        this.nextSendAt = Date.now();
      }

      this.sending = true;
      atlas.send(slice, { binary: true, compress: false }, (error) => {
        this.sending = false;
        if (this.closed) return;

        // Only remove bytes after ws has accepted the complete frame. If the
        // current Atlas socket dies, retain the frame and retry after attach.
        if (error) {
          console.warn(`Atlas PCM send paused: ${error.message}`);
          this.schedule(50);
          return;
        }

        this.buffer = this.buffer.subarray(sendLength);
        this.nextSendAt += PCM_SLICE_MS;
        const now = Date.now();
        if (this.nextSendAt < now - PCM_SLICE_MS * 2) {
          this.nextSendAt = now;
        }
        this.schedule(Math.max(0, this.nextSendAt - now));
      });
      return;
    }

    if (this.done && this.buffer.length === 0) {
      if (atlas.bufferedAmount > 0) {
        this.schedule(5);
        return;
      }
      this.closed = true;
      this.onDrained();
      return;
    }

    // Gemini often emits audio in irregular chunk sizes. Keep the single
    // 20 ms output clock alive without inserting silence into the ESP32 ring.
    this.schedule(2);
  }
}


const server = http.createServer(async (req, res) => {
  let parsed;
  try {
    parsed = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  } catch {
    jsonResponse(res, 400, { ok: false, error: "bad_url" });
    return;
  }

  if (req.method === "GET" && (parsed.pathname === "/" || parsed.pathname === "/health")) {
    jsonResponse(res, 200, {
      ok: true,
      service: "atlas-live-relay",
      version: RELAY_VERSION,
      model: GEMINI_PRIMARY_MODEL,
      fallbackModel: GEMINI_FALLBACK_MODEL,
      voice: ATLAS_VOICE,
      mode: "persistent-device-bridge-with-gemini-resumption",
      geminiTransport: "official-google-genai-sdk",
      uplink: "captured-pcm-over-persistent-wss",
      downlink: "20ms-clocked-pcm-esp32-jitter-buffer",
      durableMemory: SUPABASE_ENABLED,
    });
    return;
  }

  if (req.method === "GET" && parsed.pathname === "/api/chats") {
    if (!authorized(parsed)) {
      jsonResponse(res, 401, { ok: false, error: "unauthorized" });
      return;
    }
    if (!SUPABASE_ENABLED) {
      jsonResponse(res, 503, { ok: false, error: "supabase_not_configured" });
      return;
    }
    try {
      const deviceId = safeDeviceId(parsed.searchParams.get("device"));
      const turns = await supabaseRequest(
        `/rest/v1/atlas_turns?device_id=eq.${encodeURIComponent(deviceId)}` +
          "&select=conversation_id,turn_index,user_text,atlas_text,created_at&order=created_at.desc&limit=100",
      );
      jsonResponse(res, 200, { ok: true, deviceId, turns });
    } catch (error) {
      jsonResponse(res, 500, { ok: false, error: error.message });
    }
    return;
  }

  if (parsed.pathname === "/api/memories") {
    if (!authorized(parsed)) {
      jsonResponse(res, 401, { ok: false, error: "unauthorized" });
      return;
    }
    if (!SUPABASE_ENABLED) {
      jsonResponse(res, 503, { ok: false, error: "supabase_not_configured" });
      return;
    }

    const deviceId = safeDeviceId(parsed.searchParams.get("device"));
    try {
      if (req.method === "GET") {
        const memories = await loadMemories(deviceId);
        jsonResponse(res, 200, { ok: true, deviceId, memories });
        return;
      }

      if (req.method === "POST") {
        const body = await readJsonBody(req);
        const content = String(body.content || "").trim();
        if (!content) {
          jsonResponse(res, 400, { ok: false, error: "content_required" });
          return;
        }
        const memoryKey = String(body.memoryKey || body.memory_key || crypto.randomUUID())
          .replace(/[^a-zA-Z0-9_.-]/g, "-")
          .slice(0, 120);
        const importance = Math.max(1, Math.min(10, Number(body.importance || 5)));
        const rows = await supabaseRequest("/rest/v1/atlas_memories?on_conflict=device_id,memory_key", {
          method: "POST",
          prefer: "resolution=merge-duplicates,return=representation",
          body: {
            device_id: deviceId,
            memory_key: memoryKey,
            content: content.slice(0, 4000),
            importance,
            source: String(body.source || "manual").slice(0, 60),
            active: body.active !== false,
            updated_at: new Date().toISOString(),
          },
        });
        jsonResponse(res, 200, { ok: true, deviceId, memory: rows?.[0] || null });
        return;
      }

      if (req.method === "DELETE") {
        const memoryKey = String(parsed.searchParams.get("memoryKey") || "").trim();
        if (!memoryKey) {
          jsonResponse(res, 400, { ok: false, error: "memoryKey_required" });
          return;
        }
        await supabaseRequest(
          `/rest/v1/atlas_memories?device_id=eq.${encodeURIComponent(deviceId)}` +
            `&memory_key=eq.${encodeURIComponent(memoryKey)}`,
          { method: "DELETE", prefer: "return=minimal" },
        );
        jsonResponse(res, 200, { ok: true, deviceId, deleted: memoryKey });
        return;
      }
    } catch (error) {
      jsonResponse(res, error.status || 500, { ok: false, error: error.message });
      return;
    }
  }

  jsonResponse(res, 404, { ok: false, error: "not_found" });
});

const atlasWss = new WebSocketServer({
  noServer: true,
  maxPayload: MAX_ATLAS_MESSAGE_BYTES,
  perMessageDeflate: false,
});

server.on("upgrade", (req, socket, head) => {
  let parsed;
  try {
    parsed = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  } catch {
    socket.destroy();
    return;
  }

  if (parsed.pathname !== "/atlas") {
    socket.destroy();
    return;
  }
  if (!authorized(parsed)) {
    socket.write("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
    socket.destroy();
    return;
  }

  req.atlasDeviceId = safeDeviceId(parsed.searchParams.get("device"));
  atlasWss.handleUpgrade(req, socket, head, (ws) => atlasWss.emit("connection", ws, req));
});


const deviceBridges = new Map();

class DeviceBridge {
  constructor(deviceId) {
    this.deviceId = deviceId;
    this.atlas = null;
    this.atlasHeartbeat = null;
    this.atlasLastSeenAt = Date.now();
    this.atlasCleanupTimer = null;

    this.geminiSession = null;
    this.geminiReady = false;
    this.geminiConnecting = false;
    this.geminiRetryTimer = null;
    this.geminiConnectionSerial = 0;
    this.intentionalGeminiCloseSerial = 0;
    this.pendingGeminiRotation = false;
    this.pendingGeminiRotationReason = "";
    this.resumeHandle = null;
    this.resumeHandleUpdatedAt = 0;

    this.conversationId = null;
    this.recentTurns = [];
    this.memories = [];
    this.turnCounter = 0;
    this.currentTurn = null;
    this.pacer = null;
    this.initialized = false;
    this.destroyed = false;

    this.activeGeminiModel = GEMINI_PRIMARY_MODEL;
    this.activeModelIndex = 0;
    this.geminiModels = [...new Set(
      [GEMINI_PRIMARY_MODEL, GEMINI_FALLBACK_MODEL].filter(Boolean),
    )];
  }

  currentAtlas() {
    return this.atlas?.readyState === WebSocket.OPEN ? this.atlas : null;
  }

  sendAtlasJson(payload) {
    const atlas = this.currentAtlas();
    if (!atlas) return false;
    try {
      atlas.send(JSON.stringify(payload), { compress: false }, (error) => {
        if (error) {
          console.warn(`[${this.deviceId}] Atlas JSON send failed: ${error.message}`);
        }
      });
      return true;
    } catch (error) {
      console.warn(`[${this.deviceId}] Atlas JSON send threw: ${error.message}`);
      return false;
    }
  }

  sendReady(resumed = false) {
    if (!this.geminiReady) return;
    this.sendAtlasJson({
      type: "ready",
      model: this.activeGeminiModel,
      voice: ATLAS_VOICE,
      conversationId: this.conversationId,
      resumed,
      durableMemory: SUPABASE_ENABLED,
      memoryCount: this.memories.length,
    });
  }

  attachAtlas(atlas) {
    if (this.destroyed) {
      atlas.close(1011, "device_bridge_closed");
      return;
    }

    const previous = this.atlas;
    if (previous && previous !== atlas && previous.readyState === WebSocket.OPEN) {
      try {
        previous.close(4001, "newer_atlas_connection");
      } catch {}
    }

    this.atlas = atlas;
    this.atlasLastSeenAt = Date.now();
    if (this.atlasCleanupTimer) clearTimeout(this.atlasCleanupTimer);
    this.atlasCleanupTimer = null;

    atlas.on("message", (data, isBinary) => {
      if (this.atlas !== atlas) return;
      this.atlasLastSeenAt = Date.now();
      this.handleAtlasMessage(data, isBinary);
    });

    atlas.on("pong", () => {
      if (this.atlas === atlas) this.atlasLastSeenAt = Date.now();
    });

    atlas.on("error", (error) => {
      if (this.atlas === atlas) {
        console.error(`[${this.deviceId}] Atlas socket error: ${error.message}`);
      }
    });

    atlas.on("close", (code, reason) => {
      if (this.atlas !== atlas) return;
      console.log(
        `[${this.deviceId}] Atlas transport detached code=${code} ` +
          `reason=${reason?.toString() || "<empty>"}; Gemini remains alive.`,
      );
      this.atlas = null;
      this.stopAtlasHeartbeat();
      this.scheduleIdleCleanup();
    });

    this.startAtlasHeartbeat();
    this.sendAtlasJson({
      type: "relay_connected",
      version: RELAY_VERSION,
      durableMemory: SUPABASE_ENABLED,
      geminiTransport: "official-google-genai-sdk",
      persistentDeviceBridge: true,
    });

    if (this.geminiReady) this.sendReady(true);
    this.pacer?.wake();

    console.log(
      `[${this.deviceId}] Atlas transport attached; ` +
        `Gemini=${this.geminiReady ? "already-ready" : "connecting"}.`,
    );
  }

  startAtlasHeartbeat() {
    this.stopAtlasHeartbeat();
    this.atlasHeartbeat = setInterval(() => {
      const atlas = this.currentAtlas();
      if (!atlas) return;

      const idleMs = Date.now() - this.atlasLastSeenAt;
      if (idleMs > ATLAS_STALE_TIMEOUT_MS) {
        console.warn(
          `[${this.deviceId}] Atlas transport silent for ${idleMs} ms; ` +
            "terminating only the stale device socket.",
        );
        atlas.terminate();
        return;
      }

      try {
        atlas.ping();
      } catch (error) {
        console.warn(`[${this.deviceId}] Atlas ping failed: ${error.message}`);
      }
    }, ATLAS_HEARTBEAT_INTERVAL_MS);
  }

  stopAtlasHeartbeat() {
    if (this.atlasHeartbeat) clearInterval(this.atlasHeartbeat);
    this.atlasHeartbeat = null;
  }

  scheduleIdleCleanup() {
    if (this.atlasCleanupTimer || this.destroyed) return;
    this.atlasCleanupTimer = setTimeout(() => {
      this.atlasCleanupTimer = null;
      if (!this.currentAtlas()) this.destroy("atlas_reattach_grace_expired");
    }, ATLAS_REATTACH_GRACE_MS);
  }

  async initialize() {
    if (this.initialized || this.destroyed) return;
    this.initialized = true;

    try {
      const [state, turns, storedMemories] = await Promise.all([
        loadSessionState(this.deviceId),
        loadRecentTurns(this.deviceId),
        loadMemories(this.deviceId),
      ]);

      this.recentTurns = turns;
      this.memories = storedMemories;
      this.conversationId =
        state?.conversation_id || (await createConversation(this.deviceId));

      const stateAge = state?.updated_at
        ? Date.now() - Date.parse(state.updated_at)
        : Number.POSITIVE_INFINITY;
      if (
        state?.resume_handle &&
        Number.isFinite(stateAge) &&
        stateAge <= SESSION_HANDLE_MAX_AGE_MS
      ) {
        this.resumeHandle = state.resume_handle;
        this.resumeHandleUpdatedAt = Date.now() - stateAge;
      }

      await this.connectGemini(0, "initial persistent session");
    } catch (error) {
      console.error(
        `[${this.deviceId}] bridge initialization failed: ${error.message}`,
      );
      this.sendAtlasJson({
        type: "error",
        source: "relay",
        message: error.message,
      });
      this.scheduleGeminiConnect(0, 5000, "bridge initialization retry");
    }
  }

  resetTurn() {
    this.currentTurn = {
      id: crypto.randomUUID(),
      deviceId: this.deviceId,
      conversationId: this.conversationId,
      turnIndex: ++this.turnCounter,
      startedAt: Date.now(),
      activityEndedAt: 0,
      inputTranscript: "",
      outputTranscript: "",
      inputBytes: 0,
      outputBytes: 0,
      firstAudioLatencyMs: 0,
      totalTurnMs: 0,
      usage: {},
      generationCompleteForwarded: false,
      turnCompleteSeen: false,
      active: true,
    };
  }

  finishTurn() {
    if (!this.currentTurn?.active) return;

    this.currentTurn.active = false;
    this.currentTurn.totalTurnMs = Date.now() - this.currentTurn.startedAt;
    const savedTurn = { ...this.currentTurn };

    setTimeout(() => {
      if (this.currentTurn?.id === savedTurn.id) {
        savedTurn.inputTranscript = this.currentTurn.inputTranscript;
        savedTurn.outputTranscript = this.currentTurn.outputTranscript;
      }

      console.log(
        `[${this.deviceId}] turn ${savedTurn.turnIndex}: ` +
          `in=${savedTurn.inputBytes} out=${savedTurn.outputBytes} ` +
          `firstAudio=${savedTurn.firstAudioLatencyMs}ms ` +
          `total=${savedTurn.totalTurnMs}ms`,
      );
      console.log(
        `[${this.deviceId}] YOU: ${savedTurn.inputTranscript || "<empty>"}`,
      );
      console.log(
        `[${this.deviceId}] ATLAS: ${savedTurn.outputTranscript || "<empty>"}`,
      );

      void saveTurn(savedTurn).catch((error) =>
        console.error("Save turn failed:", error.message),
      );
    }, TRANSCRIPT_SETTLE_MS);
  }

  forwardGenerationComplete() {
    if (!this.currentTurn?.active || this.currentTurn.generationCompleteForwarded) {
      return;
    }
    this.currentTurn.generationCompleteForwarded = true;
    this.sendAtlasJson({ type: "generation_complete" });
  }

  completeTurnAfterPcmDrain() {
    this.forwardGenerationComplete();
    this.sendAtlasJson({ type: "turn_complete" });
    this.pacer = null;
    this.finishTurn();
    this.rotateGeminiIfSafe();
  }

  handleTurnComplete() {
    if (!this.currentTurn?.active || this.currentTurn.turnCompleteSeen) return;
    this.currentTurn.turnCompleteSeen = true;
    this.forwardGenerationComplete();

    if (!this.pacer) {
      this.sendAtlasJson({ type: "turn_complete" });
      this.finishTurn();
      this.rotateGeminiIfSafe();
      return;
    }

    this.pacer.markDone();
  }

  handleAtlasMessage(data, isBinary) {
    if (isBinary) {
      if (!this.geminiReady || !this.geminiSession || !this.currentTurn?.active) {
        return;
      }

      const audio = Buffer.from(data);
      this.currentTurn.inputBytes += audio.length;
      try {
        this.geminiSession.sendRealtimeInput({
          audio: {
            data: audio.toString("base64"),
            mimeType: "audio/pcm;rate=16000",
          },
        });
      } catch (error) {
        console.error(
          `[${this.deviceId}] Gemini audio send failed: ${error.message}`,
        );
      }
      return;
    }

    let control;
    try {
      control = JSON.parse(data.toString());
    } catch {
      return;
    }

    if (control.type === "ping") {
      this.sendAtlasJson({ type: "pong", at: Date.now() });
      return;
    }

    if (!this.geminiReady || !this.geminiSession) {
      this.sendAtlasJson({
        type: "session_reconnecting",
        reason: "gemini_not_ready",
      });
      return;
    }

    if (control.type === "activity_start") {
      this.pacer?.clear();
      this.pacer = null;
      this.resetTurn();

      try {
        this.geminiSession.sendRealtimeInput({ activityStart: {} });
        this.sendAtlasJson({
          type: "activity_started",
          turnIndex: this.currentTurn.turnIndex,
        });
      } catch (error) {
        console.error(
          `[${this.deviceId}] activityStart failed: ${error.message}`,
        );
      }
      return;
    }

    if (control.type === "activity_end" && this.currentTurn?.active) {
      this.currentTurn.activityEndedAt = Date.now();
      try {
        this.geminiSession.sendRealtimeInput({ activityEnd: {} });
        this.sendAtlasJson({
          type: "activity_ended",
          inputBytes: this.currentTurn.inputBytes,
        });
      } catch (error) {
        console.error(
          `[${this.deviceId}] activityEnd failed: ${error.message}`,
        );
      }
    }
  }

  handleGeminiMessage(message, connectionSerial) {
    if (
      this.destroyed ||
      connectionSerial !== this.geminiConnectionSerial ||
      !message
    ) {
      return;
    }

    const resumption = message.sessionResumptionUpdate;
    if (resumption?.resumable && resumption.newHandle) {
      this.resumeHandle = resumption.newHandle;
      this.resumeHandleUpdatedAt = Date.now();
      void upsertSessionState(
        this.deviceId,
        this.conversationId,
        this.resumeHandle,
      ).catch((error) =>
        console.warn(`[${this.deviceId}] Resume handle save failed: ${error.message}`),
      );
    }

    const content = message.serverContent;
    if (content) {
      if (content.interrupted) {
        this.pacer?.clear();
        this.pacer = null;
        this.sendAtlasJson({ type: "interrupted" });
        this.finishTurn();
      }

      if (content.inputTranscription?.text && this.currentTurn) {
        this.currentTurn.inputTranscript = mergeTranscript(
          this.currentTurn.inputTranscript,
          content.inputTranscription.text,
        );
        this.sendAtlasJson({
          type: "input_transcript",
          text: content.inputTranscription.text,
        });
      }

      if (content.outputTranscription?.text && this.currentTurn) {
        this.currentTurn.outputTranscript = mergeTranscript(
          this.currentTurn.outputTranscript,
          content.outputTranscription.text,
        );
        this.sendAtlasJson({
          type: "output_transcript",
          text: content.outputTranscription.text,
        });
      }

      const parts = content.modelTurn?.parts || [];
      for (const part of parts) {
        const inline = part.inlineData;
        if (!inline?.data || !(inline.mimeType || "").startsWith("audio/pcm")) {
          continue;
        }

        const pcm = Buffer.from(inline.data, "base64");
        if (!this.currentTurn) this.resetTurn();

        if (!this.currentTurn.firstAudioLatencyMs) {
          this.currentTurn.firstAudioLatencyMs =
            Date.now() -
            (this.currentTurn.activityEndedAt || this.currentTurn.startedAt);
          this.sendAtlasJson({
            type: "response_start",
            latencyMs: this.currentTurn.firstAudioLatencyMs,
          });
        }

        this.currentTurn.outputBytes += pcm.length;

        if (!this.pacer || this.pacer.closed) {
          this.pacer = new PcmPacer(
            () => this.currentAtlas(),
            () => this.completeTurnAfterPcmDrain(),
          );
        }
        this.pacer.enqueue(pcm);
      }

      if (content.generationComplete) {
        console.log(`[${this.deviceId}] Gemini generationComplete received.`);
        this.forwardGenerationComplete();
      }

      if (content.turnComplete) {
        console.log(`[${this.deviceId}] Gemini turnComplete received.`);
        this.handleTurnComplete();
      }
    }

    if (message.usageMetadata && this.currentTurn) {
      this.currentTurn.usage = message.usageMetadata;
    }

    if (message.goAway) {
      console.log(
        `[${this.deviceId}] Gemini GoAway received; preserving the session ` +
          "through resumption after the current turn.",
      );
      this.sendAtlasJson({ type: "go_away", detail: message.goAway });
      this.pendingGeminiRotation = true;
      this.pendingGeminiRotationReason = "planned Gemini connection rotation";
      this.rotateGeminiIfSafe();
    }
  }

  scheduleGeminiConnect(modelIndex, delayMs, reason) {
    if (this.destroyed) return;
    if (this.geminiRetryTimer) clearTimeout(this.geminiRetryTimer);

    if (reason) {
      console.log(
        `[${this.deviceId}] Gemini connect scheduled in ${delayMs} ms: ${reason}`,
      );
    }

    this.geminiRetryTimer = setTimeout(() => {
      this.geminiRetryTimer = null;
      void this.connectGemini(modelIndex, reason);
    }, delayMs);
  }

  closeGeminiIntentionally() {
    const session = this.geminiSession;
    this.geminiSession = null;
    this.geminiReady = false;
    this.geminiConnecting = false;
    this.intentionalGeminiCloseSerial = ++this.geminiConnectionSerial;
    if (session) {
      try {
        session.close();
      } catch (error) {
        console.warn(
          `[${this.deviceId}] Gemini close warning: ${error.message}`,
        );
      }
    }
  }

  rotateGeminiIfSafe() {
    if (!this.pendingGeminiRotation || this.destroyed) return;
    if (this.currentTurn?.active) return;
    if (this.pacer && !this.pacer.closed) return;

    const reason =
      this.pendingGeminiRotationReason || "deferred Gemini rotation";
    this.pendingGeminiRotation = false;
    this.pendingGeminiRotationReason = "";

    console.log(
      `[${this.deviceId}] Rotating Gemini connection with saved resume handle.`,
    );
    this.closeGeminiIntentionally();
    this.scheduleGeminiConnect(0, 100, reason);
  }

  async connectGemini(requestedModelIndex = 0, reason = "connect") {
    if (this.destroyed || this.geminiConnecting || this.geminiReady) return;

    this.activeModelIndex = Math.max(
      0,
      Math.min(requestedModelIndex, this.geminiModels.length - 1),
    );
    this.activeGeminiModel = this.geminiModels[this.activeModelIndex];

    const connectionSerial = ++this.geminiConnectionSerial;
    const resumeHandleUsed = this.resumeHandle || null;
    this.geminiConnecting = true;

    console.log(
      `[${this.deviceId}] Opening persistent Gemini Live ` +
        `model=${this.activeGeminiModel} ` +
        `resume=${resumeHandleUsed ? "yes" : "no"} reason=${reason}.`,
    );

    const config = {
      responseModalities: [Modality.AUDIO],
      thinkingConfig: { thinkingLevel: "minimal" },
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName: ATLAS_VOICE },
        },
      },
      systemInstruction: {
        parts: [{ text: buildSystemInstruction(this.memories, this.recentTurns) }],
      },
      realtimeInputConfig: {
        automaticActivityDetection: { disabled: true },
      },
      inputAudioTranscription: {},
      outputAudioTranscription: {},
      contextWindowCompression: { slidingWindow: {} },
      sessionResumption: {
        handle: resumeHandleUsed || undefined,
      },
    };

    try {
      const session = await googleAi.live.connect({
        model: this.activeGeminiModel,
        config,
        callbacks: {
          onopen: () => {
            console.log(
              `[${this.deviceId}] Gemini SDK socket opened ` +
                `model=${this.activeGeminiModel}.`,
            );
          },

          onmessage: (message) => {
            this.handleGeminiMessage(message, connectionSerial);
          },

          onerror: (event) => {
            if (connectionSerial !== this.geminiConnectionSerial) return;
            const detail =
              event?.message ||
              event?.error?.message ||
              String(event || "unknown Gemini SDK error");
            console.error(
              `[${this.deviceId}] Gemini SDK error ` +
                `model=${this.activeGeminiModel}: ${detail}`,
            );
          },

          onclose: (event) => {
            if (connectionSerial !== this.geminiConnectionSerial) return;

            const wasReady = this.geminiReady;
            this.geminiSession = null;
            this.geminiReady = false;
            this.geminiConnecting = false;

            const code = event?.code ?? 0;
            const closeReason = event?.reason || "<empty>";
            console.log(
              `[${this.deviceId}] Gemini SDK closed code=${code} ` +
                `model=${this.activeGeminiModel} reason=${closeReason}.`,
            );

            if (this.destroyed || !wasReady) return;

            if (this.currentTurn?.active && this.pacer && !this.pacer.closed) {
              // Preserve every PCM byte already received. A resumed Gemini
              // connection is opened after the device drains that audio.
              this.pendingGeminiRotation = true;
              this.pendingGeminiRotationReason = "Gemini closed after reply audio";
              this.pacer.markDone();
              return;
            }

            this.scheduleGeminiConnect(
              0,
              350,
              "persistent Gemini connection closed",
            );
          },
        },
      });

      if (
        this.destroyed ||
        connectionSerial !== this.geminiConnectionSerial
      ) {
        try {
          session.close();
        } catch {}
        return;
      }

      this.geminiSession = session;
      this.geminiConnecting = false;
      this.geminiReady = true;
      this.sendReady(Boolean(resumeHandleUsed));

      console.log(
        `[${this.deviceId}] Gemini Live ready model=${this.activeGeminiModel} ` +
          `resumed=${Boolean(resumeHandleUsed)}.`,
      );
    } catch (error) {
      if (connectionSerial !== this.geminiConnectionSerial || this.destroyed) {
        return;
      }

      this.geminiSession = null;
      this.geminiReady = false;
      this.geminiConnecting = false;

      console.error(
        `[${this.deviceId}] Gemini connect failed ` +
          `model=${this.activeGeminiModel}: ${error.message}`,
      );

      // A stale/invalid handle must never strand Atlas. Retry the same model
      // once without it, then use the configured fallback model.
      if (resumeHandleUsed) {
        this.resumeHandle = null;
        this.resumeHandleUpdatedAt = 0;
        void clearSessionState(this.deviceId).catch(() => {});
        this.scheduleGeminiConnect(
          this.activeModelIndex,
          250,
          "resume handle rejected; opening fresh connection",
        );
        return;
      }

      const nextModelIndex = this.activeModelIndex + 1;
      if (nextModelIndex < this.geminiModels.length) {
        this.scheduleGeminiConnect(
          nextModelIndex,
          350,
          `model connection failed: ${error.message}`,
        );
        return;
      }

      this.sendAtlasJson({
        type: "error",
        source: "gemini",
        message:
          "Gemini Live setup failed on all configured models. " +
          `Last error: ${error.message}`,
      });
      this.scheduleGeminiConnect(0, 5000, "all Gemini models failed");
    }
  }

  destroy(reason) {
    if (this.destroyed) return;
    this.destroyed = true;
    console.log(`[${this.deviceId}] Destroying device bridge: ${reason}.`);

    this.stopAtlasHeartbeat();
    if (this.atlasCleanupTimer) clearTimeout(this.atlasCleanupTimer);
    if (this.geminiRetryTimer) clearTimeout(this.geminiRetryTimer);
    this.atlasCleanupTimer = null;
    this.geminiRetryTimer = null;

    this.pacer?.clear();
    this.pacer = null;
    this.closeGeminiIntentionally();

    const atlas = this.atlas;
    this.atlas = null;
    if (atlas && atlas.readyState === WebSocket.OPEN) {
      try {
        atlas.close(1000, "device_bridge_closed");
      } catch {}
    }

    deviceBridges.delete(this.deviceId);
  }
}

atlasWss.on("connection", (atlas, req) => {
  const deviceId = req.atlasDeviceId || "atlas-v1";
  let bridge = deviceBridges.get(deviceId);

  if (!bridge || bridge.destroyed) {
    bridge = new DeviceBridge(deviceId);
    deviceBridges.set(deviceId, bridge);
    bridge.attachAtlas(atlas);
    void bridge.initialize();
    return;
  }

  bridge.attachAtlas(atlas);
});

server.keepAliveTimeout = 65_000;
server.headersTimeout = 70_000;
server.requestTimeout = 0;

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Atlas Live Relay ${RELAY_VERSION} listening on port ${PORT}`);
  console.log("Gemini transport: official @google/genai Live SDK");
  console.log(`Primary model: ${GEMINI_PRIMARY_MODEL}`);
  console.log(`Fallback model: ${GEMINI_FALLBACK_MODEL}`);
  console.log(`Voice: ${ATLAS_VOICE}`);
  console.log(`Durable memory: ${SUPABASE_ENABLED ? "enabled" : "disabled"}`);
});
