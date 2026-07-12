import http from "node:http";
import crypto from "node:crypto";
import { URL } from "node:url";
import { GoogleGenAI, Modality } from "@google/genai";
import { WebSocket, WebSocketServer } from "ws";

const PORT = Number(process.env.PORT || 3000);
const RELAY_VERSION = "4.0.0-button-toggle";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const ATLAS_DEVICE_TOKEN = process.env.ATLAS_DEVICE_TOKEN || "";
const GEMINI_PRIMARY_MODEL =
  process.env.GEMINI_MODEL || "gemini-3.1-flash-live-preview";
const GEMINI_FALLBACK_MODEL =
  process.env.GEMINI_FALLBACK_MODEL ||
  "gemini-2.5-flash-native-audio-preview-12-2025";
const GEMINI_RETRY_DELAY_MS = 1500;
const ATLAS_VOICE = "Charon";
const BASE_SYSTEM_INSTRUCTION =
  process.env.ATLAS_SYSTEM_INSTRUCTION ||
  "You are Atlas, a fast, practical AI companion. Speak naturally and directly. " +
    "Keep routine replies concise. Ask one useful follow-up only when it is genuinely needed.";

const LANGUAGE_AND_VOICE_LOCK =
  "LANGUAGE AND VOICE REQUIREMENT: Always speak in English only. " +
  "Never answer in Spanish or any other language, even if automatic transcription " +
  "mistakenly labels the owner's English speech as another language. " +
  "Speak as an articulate adult British man using cultivated modern Received " +
  "Pronunciation: sophisticated, measured, intelligent, warm, and natural. " +
  "Avoid American pronunciation, exaggerated aristocratic affectation, theatrical " +
  "delivery, rushed speech, and vocal fry. Use clean phrasing and an even pace. " +
  "Do not imitate the language or accent inferred from noisy audio. " +
  "Only change language if the owner explicitly says the exact words: switch language.";

const SUPABASE_URL = (process.env.SUPABASE_URL || "").replace(/\/$/, "");
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || "";
const SUPABASE_ENABLED = Boolean(SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY);

const MAX_ATLAS_MESSAGE_BYTES = 64 * 1024;
const PCM_SLICE_BYTES = 1920; // 40 ms of 24 kHz mono PCM16.
const PCM_SLICE_MS = 40;
const PCM_START_BUFFER_BYTES = 19_200; // 400 ms before first downlink frame.
const ATLAS_BACKPRESSURE_BYTES = 128 * 1024;
const SESSION_HANDLE_MAX_AGE_MS = 90 * 60 * 1000;
const RECENT_HISTORY_TURNS = 4;
const TRANSCRIPT_SETTLE_MS = 450;

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

class PcmChunkQueue {
  constructor() {
    this.chunks = [];
    this.headOffset = 0;
    this.length = 0;
  }

  push(pcm) {
    if (!pcm?.length) return;
    const chunk = Buffer.from(pcm);
    this.chunks.push(chunk);
    this.length += chunk.length;
  }

  take(byteCount) {
    if (byteCount <= 0 || this.length < byteCount) return null;

    const output = Buffer.allocUnsafe(byteCount);
    let outputOffset = 0;

    while (outputOffset < byteCount) {
      const head = this.chunks[0];
      const available = head.length - this.headOffset;
      const wanted = byteCount - outputOffset;
      const copyCount = available < wanted ? available : wanted;

      head.copy(
        output,
        outputOffset,
        this.headOffset,
        this.headOffset + copyCount,
      );

      outputOffset += copyCount;
      this.headOffset += copyCount;
      this.length -= copyCount;

      if (this.headOffset >= head.length) {
        this.chunks.shift();
        this.headOffset = 0;
      }
    }

    return output;
  }

  takeAllEven() {
    const evenLength = this.length & ~1;
    if (!evenLength) return null;
    return this.take(evenLength);
  }

  clear() {
    this.chunks = [];
    this.headOffset = 0;
    this.length = 0;
  }
}

class PcmPacer {
  constructor(atlas, onDrained) {
    this.atlas = atlas;
    this.onDrained = onDrained;
    this.queue = new PcmChunkQueue();
    this.timer = null;
    this.sending = false;
    this.done = false;
    this.closed = false;
    this.started = false;
    this.nextSendAt = 0;
    this.framesSent = 0;
    this.minSendGapMs = Number.POSITIVE_INFINITY;
    this.maxSendGapMs = 0;
    this.lastSendStartedAt = 0;
    this.backpressureWaits = 0;
  }

  enqueue(pcm) {
    if (this.closed || !pcm?.length) return;
    this.queue.push(pcm);

    if (
      !this.timer &&
      !this.sending &&
      (this.started || this.done || this.queue.length >= PCM_START_BUFFER_BYTES)
    ) {
      this.schedule(0);
    }
  }

  markDone() {
    this.done = true;
    if (!this.timer && !this.sending) this.schedule(0);
  }

  clear() {
    this.closed = true;
    this.queue.clear();
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.sending = false;
  }

  schedule(delayMs) {
    if (this.closed || this.timer || this.sending) return;
    this.timer = setTimeout(
      () => this.tick(),
      Math.max(0, Math.ceil(delayMs)),
    );
  }

  tick() {
    this.timer = null;
    if (this.closed || this.sending) return;

    if (this.atlas.readyState !== WebSocket.OPEN) {
      this.clear();
      return;
    }

    if (!this.started) {
      if (!this.done && this.queue.length < PCM_START_BUFFER_BYTES) return;
      this.started = true;
      this.nextSendAt = Date.now();
    }

    if (this.atlas.bufferedAmount > ATLAS_BACKPRESSURE_BYTES) {
      this.backpressureWaits++;
      this.schedule(5);
      return;
    }

    if (this.queue.length >= PCM_SLICE_BYTES) {
      const now = Date.now();
      if (now < this.nextSendAt) {
        this.schedule(this.nextSendAt - now);
        return;
      }

      const slice = this.queue.take(PCM_SLICE_BYTES);
      this.sending = true;
      const sendStartedAt = Date.now();

      if (this.lastSendStartedAt > 0) {
        const gapMs = sendStartedAt - this.lastSendStartedAt;
        this.minSendGapMs = Math.min(this.minSendGapMs, gapMs);
        this.maxSendGapMs = Math.max(this.maxSendGapMs, gapMs);
      }
      this.lastSendStartedAt = sendStartedAt;
      this.framesSent++;

      try {
        this.atlas.send(slice, { binary: true }, (error) => {
          this.sending = false;
          if (this.closed) return;

          if (error) {
            console.error('Atlas PCM send failed:', error.message);
            this.clear();
            return;
          }

          // Never catch up by firing multiple frames back-to-back. If the
          // event loop or socket was delayed, restart the 40 ms clock from
          // now. This guarantees Railway cannot outrun 24 kHz playback.
          const callbackAt = Date.now();
          const nominalNext = this.nextSendAt + PCM_SLICE_MS;
          this.nextSendAt = nominalNext <= callbackAt
            ? callbackAt + PCM_SLICE_MS
            : nominalNext;

          this.schedule(this.nextSendAt - Date.now());
        });
      } catch (error) {
        this.sending = false;
        console.error('Atlas PCM send threw:', error.message);
        this.clear();
      }
      return;
    }

    if (this.done) {
      const tail = this.queue.takeAllEven();
      this.queue.clear();

      const complete = () => {
        if (this.closed) return;
        this.closed = true;
        const minGap = Number.isFinite(this.minSendGapMs)
          ? this.minSendGapMs
          : 0;
        console.log(
          `PCM pacing complete: frames=${this.framesSent} ` +
            `minGapMs=${minGap} maxGapMs=${this.maxSendGapMs} ` +
            `backpressureWaits=${this.backpressureWaits}`,
        );
        this.onDrained();
      };

      if (tail?.length) {
        this.sending = true;
        try {
          this.atlas.send(tail, { binary: true }, (error) => {
            this.sending = false;
            if (error) {
              console.error('Atlas final PCM send failed:', error.message);
              this.clear();
              return;
            }
            complete();
          });
        } catch (error) {
          this.sending = false;
          console.error('Atlas final PCM send threw:', error.message);
          this.clear();
        }
      } else {
        complete();
      }
      return;
    }

    // Gemini has not produced another complete 40 ms frame yet.
    this.schedule(4);
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
      mode: "button-toggle-fresh-gemini-session-per-turn",
      geminiTransport: "official-google-genai-sdk",
      uplink: "record-then-flow-controlled-pcm-upload",
      downlink: "strict-one-frame-per-40ms-no-catch-up",
      transportHeartbeat: "activity-aware-120s-stale-timeout",
      turnControl: "press-start-press-stop",
      geminiSession: "fresh-per-turn",
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

const atlasWss = new WebSocketServer({ noServer: true, maxPayload: MAX_ATLAS_MESSAGE_BYTES });

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

atlasWss.on("connection", (atlas, req) => {
  const deviceId = req.atlasDeviceId || "atlas-v1";

  let closed = false;
  let conversationId = null;
  let recentTurns = [];
  let memories = [];

  let turnCounter = 0;
  let turnState = "idle";
  let currentTurn = null;
  let pacer = null;

  let geminiSession = null;
  let geminiConnectionSerial = 0;
  let activeGeminiModel = GEMINI_PRIMARY_MODEL;

  let atlasLastSeenAt = Date.now();
  let atlasTransportHeartbeat = null;
  let lastUploadAckBytes = 0;

  const geminiModels = [...new Set(
    [GEMINI_PRIMARY_MODEL, GEMINI_FALLBACK_MODEL].filter(Boolean),
  )];

  const sendAtlasJson = (payload) => {
    if (atlas.readyState === WebSocket.OPEN) {
      atlas.send(JSON.stringify(payload));
    }
  };

  const closeGeminiSession = () => {
    // Incrementing the serial makes callbacks from the old session harmless.
    geminiConnectionSerial++;
    const session = geminiSession;
    geminiSession = null;

    if (session) {
      try {
        session.close();
      } catch (error) {
        console.warn(
          `[${deviceId}] Gemini close warning: ${error.message}`,
        );
      }
    }
  };

  const resetCurrentTurn = () => {
    currentTurn = {
      id: crypto.randomUUID(),
      deviceId,
      conversationId,
      turnIndex: ++turnCounter,
      startedAt: Date.now(),
      activityEndedAt: 0,
      inputTranscript: "",
      outputTranscript: "",
      inputBytes: 0,
      outputBytes: 0,
      firstAudioLatencyMs: 0,
      totalTurnMs: 0,
      usage: {},
      active: true,
    };
    lastUploadAckBytes = 0;
  };

  const rememberCompletedTurn = (turn) => {
    if (!turn?.inputTranscript || !turn?.outputTranscript) return;

    recentTurns.push({
      user_text: turn.inputTranscript,
      atlas_text: turn.outputTranscript,
      created_at: new Date().toISOString(),
    });
    if (recentTurns.length > RECENT_HISTORY_TURNS) {
      recentTurns = recentTurns.slice(-RECENT_HISTORY_TURNS);
    }
  };

  const saveCompletedTurn = (turn) => {
    if (!turn?.active) return;

    turn.active = false;
    turn.totalTurnMs = Date.now() - turn.startedAt;
    const savedTurn = { ...turn };

    setTimeout(() => {
      // Transcription fragments can arrive shortly after the final audio.
      if (currentTurn?.id === savedTurn.id) {
        savedTurn.inputTranscript = currentTurn.inputTranscript;
        savedTurn.outputTranscript = currentTurn.outputTranscript;
      }

      console.log(
        `[${deviceId}] turn ${savedTurn.turnIndex}: ` +
          `in=${savedTurn.inputBytes} out=${savedTurn.outputBytes} ` +
          `firstAudio=${savedTurn.firstAudioLatencyMs}ms ` +
          `total=${savedTurn.totalTurnMs}ms`,
      );
      console.log(
        `[${deviceId}] YOU: ${savedTurn.inputTranscript || "<empty>"}`,
      );
      console.log(
        `[${deviceId}] ATLAS: ${savedTurn.outputTranscript || "<empty>"}`,
      );

      rememberCompletedTurn(savedTurn);

      void saveTurn(savedTurn).catch((error) =>
        console.error("Save turn failed:", error.message),
      );
    }, TRANSCRIPT_SETTLE_MS);
  };

  const becomeReadyForNextTurn = () => {
    closeGeminiSession();
    currentTurn = null;
    turnState = "idle";
    sendAtlasJson({
      type: "ready_for_next",
      mode: "button_toggle",
    });
  };

  const abortTurn = (message, detail = "") => {
    console.error(
      `[${deviceId}] Button turn aborted: ${message}` +
        (detail ? ` (${detail})` : ""),
    );

    pacer?.clear();
    pacer = null;
    closeGeminiSession();

    if (currentTurn) currentTurn.active = false;
    currentTurn = null;
    turnState = "idle";

    sendAtlasJson({
      type: "turn_error",
      message,
      detail,
    });
    sendAtlasJson({
      type: "ready_for_next",
      mode: "button_toggle",
    });
  };

  const completeTurnAfterAudio = () => {
    if (!currentTurn?.active) return;

    sendAtlasJson({ type: "generation_complete" });
    sendAtlasJson({ type: "turn_complete" });

    saveCompletedTurn(currentTurn);
    pacer = null;

    // This Live session has done its one job. Closing it here removes all
    // multi-turn session rotation and second-turn reconnect edge cases.
    becomeReadyForNextTurn();
  };

  const handleGenerationDone = () => {
    if (!currentTurn?.active) return;

    if (!pacer) {
      completeTurnAfterAudio();
      return;
    }

    pacer.markDone();
  };

  const handleGeminiMessage = (message, connectionSerial) => {
    if (
      closed ||
      connectionSerial !== geminiConnectionSerial ||
      !message ||
      !currentTurn
    ) {
      return;
    }

    const content = message.serverContent;

    if (content) {
      if (content.interrupted) {
        abortTurn("Gemini interrupted the response.");
        return;
      }

      if (content.inputTranscription?.text) {
        currentTurn.inputTranscript = mergeTranscript(
          currentTurn.inputTranscript,
          content.inputTranscription.text,
        );
        sendAtlasJson({
          type: "input_transcript",
          text: content.inputTranscription.text,
        });
      }

      if (content.outputTranscription?.text) {
        currentTurn.outputTranscript = mergeTranscript(
          currentTurn.outputTranscript,
          content.outputTranscription.text,
        );
        sendAtlasJson({
          type: "output_transcript",
          text: content.outputTranscription.text,
        });
      }

      const parts = content.modelTurn?.parts || [];
      for (const part of parts) {
        const inline = part.inlineData;
        if (
          !inline?.data ||
          !(inline.mimeType || "").startsWith("audio/pcm")
        ) {
          continue;
        }

        const pcm = Buffer.from(inline.data, "base64");

        if (!currentTurn.firstAudioLatencyMs) {
          currentTurn.firstAudioLatencyMs =
            Date.now() -
            (currentTurn.activityEndedAt || currentTurn.startedAt);

          sendAtlasJson({
            type: "response_start",
            latencyMs: currentTurn.firstAudioLatencyMs,
          });
        }

        currentTurn.outputBytes += pcm.length;
        turnState = "speaking";

        if (!pacer || pacer.closed) {
          pacer = new PcmPacer(atlas, completeTurnAfterAudio);
        }
        pacer.enqueue(pcm);
      }

      if (content.generationComplete || content.turnComplete) {
        console.log(
          `[${deviceId}] Gemini end signal: generationComplete=` +
            `${Boolean(content.generationComplete)} turnComplete=` +
            `${Boolean(content.turnComplete)}`,
        );
        handleGenerationDone();
      }
    }

    if (message.usageMetadata) {
      currentTurn.usage = message.usageMetadata;
    }

    // A fresh Gemini session is used for every button turn, so a GoAway notice
    // is merely logged. We never rotate or close it in the middle of a turn.
    if (message.goAway) {
      console.log(
        `[${deviceId}] Gemini GoAway received during short per-turn session; ` +
          "deferring close until this turn finishes.",
      );
    }
  };

  const openGeminiForTurn = async (requestedModelIndex = 0) => {
    if (
      closed ||
      turnState !== "preparing" ||
      !currentTurn?.active
    ) {
      return;
    }

    const modelIndex = Math.max(
      0,
      Math.min(requestedModelIndex, geminiModels.length - 1),
    );
    activeGeminiModel = geminiModels[modelIndex];

    const connectionSerial = ++geminiConnectionSerial;

    console.log(
      `[${deviceId}] Opening fresh Gemini session for button turn ` +
        `${currentTurn.turnIndex}; model=${activeGeminiModel}`,
    );

    const config = {
      responseModalities: [Modality.AUDIO],
      thinkingConfig: { thinkingLevel: "minimal" },
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: {
            voiceName: ATLAS_VOICE,
          },
        },
      },
      systemInstruction: {
        parts: [
          {
            text: buildSystemInstruction(memories, recentTurns),
          },
        ],
      },
      realtimeInputConfig: {
        automaticActivityDetection: {
          disabled: true,
        },
      },
      inputAudioTranscription: {},
      outputAudioTranscription: {},
    };

    try {
      const session = await googleAi.live.connect({
        model: activeGeminiModel,
        config,
        callbacks: {
          onopen: () => {
            console.log(
              `[${deviceId}] Gemini SDK socket opened for turn ` +
                `${currentTurn?.turnIndex || "?"}.`,
            );
          },

          onmessage: (message) => {
            handleGeminiMessage(message, connectionSerial);
          },

          onerror: (event) => {
            if (connectionSerial !== geminiConnectionSerial) return;

            const detail =
              event?.message ||
              event?.error?.message ||
              String(event || "unknown Gemini SDK error");

            console.error(
              `[${deviceId}] Gemini SDK error model=${activeGeminiModel}: ` +
                detail,
            );
          },

          onclose: (event) => {
            if (connectionSerial !== geminiConnectionSerial) return;
            if (closed || turnState === "idle") return;

            const code = event?.code ?? 0;
            const reason = event?.reason || "<empty>";

            abortTurn(
              `Gemini closed during the button turn (code ${code}).`,
              reason,
            );
          },
        },
      });

      if (
        closed ||
        connectionSerial !== geminiConnectionSerial ||
        turnState !== "preparing"
      ) {
        try {
          session.close();
        } catch {}
        return;
      }

      geminiSession = session;
      turnState = "ready_to_record";

      sendAtlasJson({
        type: "turn_ready",
        model: activeGeminiModel,
        voice: ATLAS_VOICE,
        turnIndex: currentTurn.turnIndex,
      });

      console.log(
        `[${deviceId}] Fresh Gemini session ready for button turn ` +
          `${currentTurn.turnIndex}.`,
      );
    } catch (error) {
      if (
        closed ||
        connectionSerial !== geminiConnectionSerial ||
        turnState !== "preparing"
      ) {
        return;
      }

      console.error(
        `[${deviceId}] Gemini connect failed model=${activeGeminiModel}: ` +
          error.message,
      );

      const nextModelIndex = modelIndex + 1;
      if (nextModelIndex < geminiModels.length) {
        await openGeminiForTurn(nextModelIndex);
        return;
      }

      abortTurn(
        "Gemini could not open a fresh session for this turn.",
        error.message,
      );
    }
  };

  const closeAll = (code = 1000, reason = "session_closed") => {
    if (closed) return;
    closed = true;

    if (atlasTransportHeartbeat) clearInterval(atlasTransportHeartbeat);
    atlasTransportHeartbeat = null;

    pacer?.clear();
    pacer = null;
    closeGeminiSession();

    if (
      atlas.readyState === WebSocket.OPEN ||
      atlas.readyState === WebSocket.CONNECTING
    ) {
      atlas.close(code, reason);
    }
  };

  atlas.on("message", (data, isBinary) => {
    atlasLastSeenAt = Date.now();

    if (isBinary) {
      if (
        turnState !== "recording" ||
        !geminiSession ||
        !currentTurn?.active
      ) {
        return;
      }

      const audio = Buffer.from(data);
      currentTurn.inputBytes += audio.length;

      try {
        geminiSession.sendRealtimeInput({
          audio: {
            data: audio.toString("base64"),
            mimeType: "audio/pcm;rate=16000",
          },
        });
      } catch (error) {
        abortTurn("Gemini audio upload failed.", error.message);
        return;
      }

      if (
        currentTurn.inputBytes - lastUploadAckBytes >= 16 * 1024
      ) {
        lastUploadAckBytes = currentTurn.inputBytes;
        sendAtlasJson({
          type: "upload_ack",
          inputBytes: currentTurn.inputBytes,
        });
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
      sendAtlasJson({ type: "pong", at: Date.now() });
      return;
    }

    if (control.type === "turn_prepare") {
      if (turnState !== "idle") {
        sendAtlasJson({
          type: "turn_busy",
          state: turnState,
        });
        return;
      }

      resetCurrentTurn();
      turnState = "preparing";
      sendAtlasJson({
        type: "turn_preparing",
        turnIndex: currentTurn.turnIndex,
      });

      void openGeminiForTurn(0);
      return;
    }

    if (control.type === "activity_start") {
      if (
        turnState !== "ready_to_record" ||
        !geminiSession ||
        !currentTurn?.active
      ) {
        sendAtlasJson({
          type: "turn_error",
          message: "Gemini turn was not ready for audio.",
        });
        return;
      }

      try {
        geminiSession.sendRealtimeInput({ activityStart: {} });
        turnState = "recording";
        sendAtlasJson({
          type: "activity_started",
          turnIndex: currentTurn.turnIndex,
        });
      } catch (error) {
        abortTurn("Could not start the Gemini audio activity.", error.message);
      }
      return;
    }

    if (control.type === "activity_end") {
      if (
        turnState !== "recording" ||
        !geminiSession ||
        !currentTurn?.active
      ) {
        sendAtlasJson({
          type: "turn_error",
          message: "No active recording was available to finish.",
        });
        return;
      }

      currentTurn.activityEndedAt = Date.now();

      try {
        geminiSession.sendRealtimeInput({ activityEnd: {} });
        turnState = "waiting";

        // Final acknowledgement covers any remainder below the 16 KB window.
        sendAtlasJson({
          type: "upload_ack",
          inputBytes: currentTurn.inputBytes,
        });
        sendAtlasJson({
          type: "activity_ended",
          inputBytes: currentTurn.inputBytes,
        });
      } catch (error) {
        abortTurn("Could not finish the Gemini audio activity.", error.message);
      }
      return;
    }

    if (control.type === "turn_cancel") {
      abortTurn("Turn cancelled by Atlas.");
    }
  });

  atlas.on("pong", () => {
    atlasLastSeenAt = Date.now();
  });

  atlasTransportHeartbeat = setInterval(() => {
    if (closed || atlas.readyState !== WebSocket.OPEN) return;

    const idleMs = Date.now() - atlasLastSeenAt;
    if (idleMs > 120_000) {
      console.warn(
        `[${deviceId}] Atlas transport silent for ${idleMs} ms; ` +
          "terminating genuinely stale socket.",
      );
      atlas.terminate();
      return;
    }

    try {
      atlas.ping();
    } catch (error) {
      console.error(`[${deviceId}] Atlas ping failed: ${error.message}`);
    }
  }, 20_000);

  atlas.on("error", (error) => {
    console.error(`[${deviceId}] Atlas socket error:`, error.message);
  });

  atlas.on("close", () => {
    console.log(`[${deviceId}] Atlas disconnected.`);
    closeAll(1000, "atlas_disconnected");
  });

  void (async () => {
    try {
      const [turns, storedMemories] = await Promise.all([
        loadRecentTurns(deviceId),
        loadMemories(deviceId),
      ]);

      recentTurns = turns.slice(-RECENT_HISTORY_TURNS);
      memories = storedMemories;
      conversationId = await createConversation(deviceId);

      sendAtlasJson({
        type: "relay_connected",
        version: RELAY_VERSION,
        durableMemory: SUPABASE_ENABLED,
        geminiTransport: "official-google-genai-sdk",
      });

      // The relay is immediately ready. Gemini is opened only after the user
      // presses the yellow button to finish a recording.
      sendAtlasJson({
        type: "relay_ready",
        mode: "button_toggle",
        voice: ATLAS_VOICE,
        durableMemory: SUPABASE_ENABLED,
        memoryCount: memories.length,
      });

      console.log(
        `[${deviceId}] Button-toggle relay ready; no Gemini session is kept ` +
          "open between turns.",
      );
    } catch (error) {
      console.error(
        `[${deviceId}] relay initialization failed:`,
        error.message,
      );
      sendAtlasJson({
        type: "error",
        source: "relay",
        message: error.message,
      });
      closeAll(1011, "relay_initialization_failed");
    }
  })();
});

server.keepAliveTimeout = 65_000;
server.headersTimeout = 70_000;
server.requestTimeout = 0;

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Atlas Live Relay ${RELAY_VERSION} listening on port ${PORT}`);
  console.log("Mode: button toggle; fresh Gemini Live session per turn");
  console.log("Gemini transport: official @google/genai Live SDK");
  console.log(`Primary model: ${GEMINI_PRIMARY_MODEL}`);
  console.log(`Fallback model: ${GEMINI_FALLBACK_MODEL}`);
  console.log(`Voice: ${ATLAS_VOICE}`);
  console.log(`Durable memory: ${SUPABASE_ENABLED ? "enabled" : "disabled"}`);
});
