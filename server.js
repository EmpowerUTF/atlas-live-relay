import http from "node:http";
import crypto from "node:crypto";
import { URL } from "node:url";
import { WebSocket, WebSocketServer } from "ws";

const PORT = Number(process.env.PORT || 3000);
const RELAY_VERSION = "3.0.1";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const ATLAS_DEVICE_TOKEN = process.env.ATLAS_DEVICE_TOKEN || "";
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.1-flash-live-preview";
const ATLAS_VOICE = process.env.ATLAS_VOICE || "Kore";
const BASE_SYSTEM_INSTRUCTION =
  process.env.ATLAS_SYSTEM_INSTRUCTION ||
  "You are Atlas, a fast, practical AI companion. Speak naturally and directly. " +
    "Keep routine replies concise. Ask one useful follow-up only when it is genuinely needed.";

const SUPABASE_URL = (process.env.SUPABASE_URL || "").replace(/\/$/, "");
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || "";
const SUPABASE_ENABLED = Boolean(SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY);

const MAX_ATLAS_MESSAGE_BYTES = 64 * 1024;
const MAX_GEMINI_MESSAGE_BYTES = 8 * 1024 * 1024;
const PCM_SLICE_BYTES = 960; // 20 ms of 24 kHz mono PCM16.
const PCM_SLICE_MS = 20;
const SESSION_HANDLE_MAX_AGE_MS = 90 * 60 * 1000;
const RECENT_HISTORY_TURNS = 10;
const TRANSCRIPT_SETTLE_MS = 450;

if (!GEMINI_API_KEY) {
  console.error("Missing required environment variable: GEMINI_API_KEY");
  process.exit(1);
}
if (!ATLAS_DEVICE_TOKEN) {
  console.error("Missing required environment variable: ATLAS_DEVICE_TOKEN");
  process.exit(1);
}

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

function buildSystemInstruction(memories) {
  if (!memories.length) return BASE_SYSTEM_INSTRUCTION;
  const memoryLines = memories.map((item) => `- ${item.content}`).join("\n");
  return (
    `${BASE_SYSTEM_INSTRUCTION}\n\n` +
    "Durable owner memory supplied by Atlas storage. Use it only when relevant; do not mention the storage system:\n" +
    memoryLines
  );
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
  constructor(atlas, onDrained) {
    this.atlas = atlas;
    this.onDrained = onDrained;
    this.buffer = Buffer.alloc(0);
    this.timer = null;
    this.done = false;
    this.closed = false;
  }

  enqueue(pcm) {
    if (this.closed || !pcm?.length) return;
    this.buffer = this.buffer.length
      ? Buffer.concat([this.buffer, pcm])
      : Buffer.from(pcm);
    if (!this.timer) this.schedule(0);
  }

  markDone() {
    this.done = true;
    if (!this.timer) this.schedule(0);
  }

  clear() {
    this.closed = true;
    this.buffer = Buffer.alloc(0);
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  schedule(delay) {
    if (this.closed) return;
    this.timer = setTimeout(() => this.tick(), delay);
  }

  tick() {
    this.timer = null;
    if (this.closed) return;

    if (this.buffer.length >= PCM_SLICE_BYTES) {
      const slice = this.buffer.subarray(0, PCM_SLICE_BYTES);
      this.buffer = this.buffer.subarray(PCM_SLICE_BYTES);
      if (this.atlas.readyState === WebSocket.OPEN) {
        this.atlas.send(slice, { binary: true });
      }
      this.schedule(PCM_SLICE_MS);
      return;
    }

    if (this.done) {
      if (this.buffer.length && this.atlas.readyState === WebSocket.OPEN) {
        this.atlas.send(this.buffer, { binary: true });
      }
      this.buffer = Buffer.alloc(0);
      this.closed = true;
      this.onDrained();
      return;
    }

    this.schedule(5);
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
      model: GEMINI_MODEL,
      voice: ATLAS_VOICE,
      mode: "persistent-websocket-manual-vad-paced-pcm",
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
  let gemini = null;
  let geminiReady = false;
  let closed = false;
  let conversationId = null;
  let resumeHandle = "";
  let recentTurns = [];
  let memories = [];
  let turnCounter = 0;
  let currentTurn = null;
  let pacer = null;
  let setupComplete = false;
  let resumed = false;
  let resumeFallbackTried = false;

  const sendAtlasJson = (payload) => {
    if (atlas.readyState === WebSocket.OPEN) atlas.send(JSON.stringify(payload));
  };

  const resetTurn = () => {
    currentTurn = {
      id: crypto.randomUUID(),
      deviceId,
      conversationId,
      turnIndex: ++turnCounter,
      startedAt: Date.now(),
      inputTranscript: "",
      outputTranscript: "",
      inputBytes: 0,
      outputBytes: 0,
      firstAudioLatencyMs: 0,
      totalTurnMs: 0,
      usage: {},
      active: true,
    };
  };

  const finishTurn = () => {
    if (!currentTurn?.active) return;
    currentTurn.active = false;
    currentTurn.totalTurnMs = Date.now() - currentTurn.startedAt;
    const savedTurn = { ...currentTurn };
    setTimeout(() => {
      // Transcription events are independent and can arrive shortly after audio completion.
      savedTurn.inputTranscript = currentTurn?.id === savedTurn.id
        ? currentTurn.inputTranscript
        : savedTurn.inputTranscript;
      savedTurn.outputTranscript = currentTurn?.id === savedTurn.id
        ? currentTurn.outputTranscript
        : savedTurn.outputTranscript;
      console.log(
        `[${deviceId}] turn ${savedTurn.turnIndex}: ` +
          `in=${savedTurn.inputBytes} out=${savedTurn.outputBytes} ` +
          `firstAudio=${savedTurn.firstAudioLatencyMs}ms total=${savedTurn.totalTurnMs}ms`,
      );
      console.log(`[${deviceId}] YOU: ${savedTurn.inputTranscript || "<empty>"}`);
      console.log(`[${deviceId}] ATLAS: ${savedTurn.outputTranscript || "<empty>"}`);
      void saveTurn(savedTurn).catch((error) => console.error("Save turn failed:", error.message));
    }, TRANSCRIPT_SETTLE_MS);
  };

  const closeAll = (code = 1000, reason = "session_closed") => {
    if (closed) return;
    closed = true;
    pacer?.clear();
    if (gemini && (gemini.readyState === WebSocket.OPEN || gemini.readyState === WebSocket.CONNECTING)) {
      gemini.close(code, reason);
    }
    if (atlas.readyState === WebSocket.OPEN || atlas.readyState === WebSocket.CONNECTING) {
      atlas.close(code, reason);
    }
  };

  const handleGenerationDone = () => {
    if (!currentTurn?.active) return;
    if (!pacer) {
      sendAtlasJson({ type: "generation_complete" });
      sendAtlasJson({ type: "turn_complete" });
      finishTurn();
      return;
    }
    pacer.markDone();
  };

  const connectGemini = () => {
    setupComplete = false;
    geminiReady = false;
    const geminiUrl =
      "wss://generativelanguage.googleapis.com/ws/" +
      "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent" +
      `?key=${encodeURIComponent(GEMINI_API_KEY)}`;

    gemini = new WebSocket(geminiUrl, {
      handshakeTimeout: 15000,
      perMessageDeflate: false,
      maxPayload: MAX_GEMINI_MESSAGE_BYTES,
    });

    gemini.on("open", () => {
      const hasHistory = !resumeHandle && recentTurns.length > 0;
      const setup = {
        model: `models/${GEMINI_MODEL}`,
        generationConfig: {
          responseModalities: ["AUDIO"],
          thinkingConfig: { thinkingLevel: "minimal" },
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: ATLAS_VOICE } },
          },
        },
        systemInstruction: {
          parts: [{ text: buildSystemInstruction(memories) }],
        },
        realtimeInputConfig: {
          automaticActivityDetection: { disabled: true },
          activityHandling: "NO_INTERRUPTION",
          turnCoverage: "TURN_INCLUDES_ONLY_ACTIVITY",
        },
        inputAudioTranscription: {},
        outputAudioTranscription: {},
        sessionResumption: resumeHandle ? { handle: resumeHandle } : {},
        contextWindowCompression: {
          triggerTokens: 24000,
          slidingWindow: { targetTokens: 12000 },
        },
      };
      if (hasHistory) setup.historyConfig = { initialHistoryInClientContent: true };
      gemini.send(JSON.stringify({ setup }));
    });

    gemini.on("message", (raw, isBinary) => {
      if (isBinary) return;
      let message;
      try {
        message = JSON.parse(raw.toString());
      } catch (error) {
        console.error("Could not parse Gemini message:", error.message);
        return;
      }

      if (message.setupComplete) {
        setupComplete = true;
        geminiReady = true;

        if (!resumeHandle && recentTurns.length > 0) {
          const turns = historyTurns(recentTurns);
          if (turns.length) {
            gemini.send(JSON.stringify({ clientContent: { turns, turnComplete: false } }));
            console.log(`[${deviceId}] seeded ${turns.length} history messages.`);
          }
        }

        sendAtlasJson({
          type: "ready",
          model: GEMINI_MODEL,
          voice: ATLAS_VOICE,
          conversationId,
          resumed,
          durableMemory: SUPABASE_ENABLED,
          memoryCount: memories.length,
        });
        console.log(`[${deviceId}] Gemini Live ready. resumed=${resumed}`);
        return;
      }

      if (message.sessionResumptionUpdate) {
        const update = message.sessionResumptionUpdate;
        if (update.resumable && update.newHandle) {
          resumeHandle = update.newHandle;
          void upsertSessionState(deviceId, conversationId, resumeHandle).catch((error) =>
            console.error("Save resume handle failed:", error.message),
          );
        }
      }

      const content = message.serverContent;
      if (content) {
        if (content.interrupted) {
          pacer?.clear();
          pacer = null;
          sendAtlasJson({ type: "interrupted" });
        }

        if (content.inputTranscription?.text && currentTurn) {
          currentTurn.inputTranscript = mergeTranscript(
            currentTurn.inputTranscript,
            content.inputTranscription.text,
          );
          sendAtlasJson({ type: "input_transcript", text: content.inputTranscription.text });
        }

        if (content.outputTranscription?.text && currentTurn) {
          currentTurn.outputTranscript = mergeTranscript(
            currentTurn.outputTranscript,
            content.outputTranscription.text,
          );
          sendAtlasJson({ type: "output_transcript", text: content.outputTranscription.text });
        }

        const parts = content.modelTurn?.parts || [];
        for (const part of parts) {
          const inline = part.inlineData;
          if (!inline?.data || !(inline.mimeType || "").startsWith("audio/pcm")) continue;
          const pcm = Buffer.from(inline.data, "base64");
          if (!currentTurn) resetTurn();
          if (!currentTurn.firstAudioLatencyMs) {
            currentTurn.firstAudioLatencyMs = Date.now() - currentTurn.startedAt;
            sendAtlasJson({
              type: "response_start",
              latencyMs: currentTurn.firstAudioLatencyMs,
            });
          }
          currentTurn.outputBytes += pcm.length;
          if (!pacer || pacer.closed) {
            pacer = new PcmPacer(atlas, () => {
              sendAtlasJson({ type: "generation_complete" });
              sendAtlasJson({ type: "turn_complete" });
              finishTurn();
              pacer = null;
            });
          }
          pacer.enqueue(pcm);
        }

        if (content.generationComplete) handleGenerationDone();
        else if (content.turnComplete && !pacer && currentTurn?.active) {
          sendAtlasJson({ type: "turn_complete" });
          finishTurn();
        }
      }

      if (message.usageMetadata && currentTurn) currentTurn.usage = message.usageMetadata;

      if (message.goAway) {
        sendAtlasJson({ type: "go_away", detail: message.goAway });
        // The ESP32 reconnects immediately; the saved resumption handle preserves context.
        setTimeout(() => closeAll(1012, "gemini_go_away"), 750);
      }
    });

    gemini.on("error", (error) => {
      console.error(`[${deviceId}] Gemini WebSocket error:`, error.message);
      sendAtlasJson({ type: "error", source: "gemini", message: error.message });
    });

    gemini.on("close", (code, reasonBuffer) => {
      const reason = reasonBuffer.toString();
      console.log(`[${deviceId}] Gemini closed: ${code} ${reason}`);

      // A stale resumption handle should not trap Atlas in a reconnect loop.
      // Retry once as a fresh Live session while keeping the ESP32 socket open.
      if (!closed && !setupComplete && resumeHandle && !resumeFallbackTried) {
        resumeFallbackTried = true;
        resumeHandle = "";
        resumed = false;
        void clearSessionState(deviceId).catch((error) =>
          console.error("Clear stale resume handle failed:", error.message),
        );
        sendAtlasJson({ type: "session_restart", reason: "resume_handle_rejected" });
        setTimeout(connectGemini, 250);
        return;
      }

      sendAtlasJson({ type: "closed", source: "gemini", code, reason });
      if (!closed) closeAll(1011, "gemini_closed");
    });
  };

  atlas.on("message", (data, isBinary) => {
    if (!geminiReady || !gemini || gemini.readyState !== WebSocket.OPEN) return;

    if (isBinary) {
      if (!currentTurn?.active) return;
      const audio = Buffer.from(data);
      currentTurn.inputBytes += audio.length;
      gemini.send(
        JSON.stringify({
          realtimeInput: {
            audio: {
              data: audio.toString("base64"),
              mimeType: "audio/pcm;rate=16000",
            },
          },
        }),
      );
      return;
    }

    let control;
    try {
      control = JSON.parse(data.toString());
    } catch {
      return;
    }

    if (control.type === "activity_start") {
      pacer?.clear();
      pacer = null;
      resetTurn();
      gemini.send(JSON.stringify({ realtimeInput: { activityStart: {} } }));
      sendAtlasJson({ type: "activity_started", turnIndex: currentTurn.turnIndex });
    } else if (control.type === "activity_end") {
      if (currentTurn?.active) {
        gemini.send(JSON.stringify({ realtimeInput: { activityEnd: {} } }));
        sendAtlasJson({ type: "activity_ended", inputBytes: currentTurn.inputBytes });
      }
    } else if (control.type === "ping") {
      sendAtlasJson({ type: "pong", at: Date.now() });
    }
  });

  atlas.on("error", (error) => console.error(`[${deviceId}] Atlas socket error:`, error.message));
  atlas.on("close", () => {
    console.log(`[${deviceId}] Atlas disconnected.`);
    closeAll(1000, "atlas_disconnected");
  });

  void (async () => {
    try {
      const [state, turns, storedMemories] = await Promise.all([
        loadSessionState(deviceId),
        loadRecentTurns(deviceId),
        loadMemories(deviceId),
      ]);
      recentTurns = turns;
      memories = storedMemories;

      const stateAge = state?.updated_at
        ? Date.now() - new Date(state.updated_at).getTime()
        : Number.POSITIVE_INFINITY;
      if (state?.resume_handle && stateAge < SESSION_HANDLE_MAX_AGE_MS) {
        resumeHandle = state.resume_handle;
        conversationId = state.conversation_id || (await createConversation(deviceId));
        resumed = true;
      } else {
        if (state?.resume_handle) await clearSessionState(deviceId);
        conversationId = await createConversation(deviceId);
      }

      sendAtlasJson({
        type: "relay_connected",
        version: RELAY_VERSION,
        durableMemory: SUPABASE_ENABLED,
      });
      connectGemini();
    } catch (error) {
      console.error(`[${deviceId}] relay initialization failed:`, error.message);
      sendAtlasJson({ type: "error", source: "relay", message: error.message });
      closeAll(1011, "relay_initialization_failed");
    }
  })();
});

server.keepAliveTimeout = 65_000;
server.headersTimeout = 70_000;
server.requestTimeout = 0;

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Atlas Live Relay ${RELAY_VERSION} listening on port ${PORT}`);
  console.log(`Model: ${GEMINI_MODEL}`);
  console.log(`Voice: ${ATLAS_VOICE}`);
  console.log(`Durable memory: ${SUPABASE_ENABLED ? "enabled" : "disabled"}`);
});
