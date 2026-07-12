import http from "node:http";
import crypto from "node:crypto";
import { URL } from "node:url";
import { WebSocket, WebSocketServer } from "ws";

const PORT = Number(process.env.PORT || 3000);
const RELAY_VERSION = "3.0.2";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const ATLAS_DEVICE_TOKEN = process.env.ATLAS_DEVICE_TOKEN || "";
const GEMINI_PRIMARY_MODEL =
  process.env.GEMINI_MODEL || "gemini-3.1-flash-live-preview";
const GEMINI_FALLBACK_MODEL =
  process.env.GEMINI_FALLBACK_MODEL ||
  "gemini-2.5-flash-native-audio-preview-12-2025";
const GEMINI_SETUP_TIMEOUT_MS = 12000;
const GEMINI_RETRY_DELAY_MS = 1200;
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
      "&select=device_id,conversation_id,resume_handle,updated_at&limit=1",
  );
  return rows?.[0] || null;
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
  let instruction = BASE_SYSTEM_INSTRUCTION;

  if (memories.length) {
    instruction +=
      "\n\nDurable owner memory supplied by Atlas storage. " +
      "Use it only when relevant; do not mention the storage system:\n" +
      memories.map((item) => `- ${item.content}`).join("\n");
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

  if (
    req.method === "GET" &&
    (parsed.pathname === "/" || parsed.pathname === "/health")
  ) {
    jsonResponse(res, 200, {
      ok: true,
      service: "atlas-live-relay",
      version: RELAY_VERSION,
      model: GEMINI_PRIMARY_MODEL,
      fallbackModel: GEMINI_FALLBACK_MODEL,
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
          "&select=conversation_id,turn_index,user_text,atlas_text,created_at" +
          "&order=created_at.desc&limit=100",
      );
      jsonResponse(res, 200, { ok: true, deviceId, turns });
    } catch (error) {
      jsonResponse(res, 500, { ok: false, error: error.message });
    }
    return;
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
  atlasWss.handleUpgrade(req, socket, head, (ws) => {
    atlasWss.emit("connection", ws, req);
  });
});

atlasWss.on("connection", (atlas, req) => {
  const deviceId = req.atlasDeviceId || "atlas-v1";

  let gemini = null;
  let geminiReady = false;
  let closed = false;
  let setupTimer = null;
  let retryTimer = null;
  let connectionSerial = 0;
  let modelIndex = 0;

  let conversationId = null;
  let recentTurns = [];
  let memories = [];
  let turnCounter = 0;
  let currentTurn = null;
  let pacer = null;

  const models = [...new Set(
    [GEMINI_PRIMARY_MODEL, GEMINI_FALLBACK_MODEL].filter(Boolean),
  )];

  const sendAtlasJson = (payload) => {
    if (atlas.readyState === WebSocket.OPEN) {
      atlas.send(JSON.stringify(payload));
    }
  };

  const resetTurn = () => {
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
  };

  const finishTurn = () => {
    if (!currentTurn?.active) return;
    currentTurn.active = false;
    currentTurn.totalTurnMs = Date.now() - currentTurn.startedAt;
    const saved = { ...currentTurn };

    setTimeout(() => {
      if (currentTurn?.id === saved.id) {
        saved.inputTranscript = currentTurn.inputTranscript;
        saved.outputTranscript = currentTurn.outputTranscript;
      }

      console.log(
        `[${deviceId}] turn ${saved.turnIndex}: ` +
          `in=${saved.inputBytes} out=${saved.outputBytes} ` +
          `firstAudio=${saved.firstAudioLatencyMs}ms total=${saved.totalTurnMs}ms`,
      );
      console.log(`[${deviceId}] YOU: ${saved.inputTranscript || "<empty>"}`);
      console.log(`[${deviceId}] ATLAS: ${saved.outputTranscript || "<empty>"}`);

      void saveTurn(saved).catch((error) =>
        console.error("Save turn failed:", error.message),
      );
    }, TRANSCRIPT_SETTLE_MS);
  };

  const closeGemini = () => {
    if (setupTimer) clearTimeout(setupTimer);
    setupTimer = null;

    const socket = gemini;
    gemini = null;
    geminiReady = false;

    if (socket) {
      try {
        socket.close(1000, "relay_reset");
      } catch {}
    }
  };

  const closeAll = (code = 1000, reason = "session_closed") => {
    if (closed) return;
    closed = true;

    if (retryTimer) clearTimeout(retryTimer);
    retryTimer = null;

    pacer?.clear();
    pacer = null;
    closeGemini();

    if (
      atlas.readyState === WebSocket.OPEN ||
      atlas.readyState === WebSocket.CONNECTING
    ) {
      atlas.close(code, reason);
    }
  };

  const scheduleReconnect = (delayMs, reason) => {
    if (closed) return;
    if (retryTimer) clearTimeout(retryTimer);

    console.log(
      `[${deviceId}] Gemini reconnect scheduled in ${delayMs} ms: ${reason}`,
    );

    retryTimer = setTimeout(() => {
      retryTimer = null;
      connectGemini(modelIndex);
    }, delayMs);
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

  const handleGeminiMessage = (message, serial) => {
    if (serial !== connectionSerial || closed || !message) return;

    if (message.setupComplete) {
      if (setupTimer) clearTimeout(setupTimer);
      setupTimer = null;
      geminiReady = true;

      sendAtlasJson({
        type: "ready",
        model: models[modelIndex],
        voice: ATLAS_VOICE,
        conversationId,
        resumed: false,
        durableMemory: SUPABASE_ENABLED,
        memoryCount: memories.length,
      });

      console.log(
        `[${deviceId}] Gemini Live ready. model=${models[modelIndex]}`,
      );
      return;
    }

    if (message.sessionResumptionUpdate?.newHandle) {
      // Version 3.0.2 deliberately does not restore raw Live handles.
    }

    if (message.goAway) {
      sendAtlasJson({ type: "go_away", detail: message.goAway });
      scheduleReconnect(GEMINI_RETRY_DELAY_MS, "Gemini GoAway");
    }

    if (message.usageMetadata && currentTurn) {
      currentTurn.usage = message.usageMetadata;
    }

    const content = message.serverContent;
    if (!content) return;

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
      sendAtlasJson({
        type: "input_transcript",
        text: content.inputTranscription.text,
      });
    }

    if (content.outputTranscription?.text && currentTurn) {
      currentTurn.outputTranscript = mergeTranscript(
        currentTurn.outputTranscript,
        content.outputTranscription.text,
      );
      sendAtlasJson({
        type: "output_transcript",
        text: content.outputTranscription.text,
      });
    }

    for (const part of content.modelTurn?.parts || []) {
      const inline = part.inlineData;
      if (
        !inline?.data ||
        !(inline.mimeType || "").startsWith("audio/pcm")
      ) {
        continue;
      }

      const pcm = Buffer.from(inline.data, "base64");
      if (!pcm.length) continue;
      if (!currentTurn) resetTurn();

      if (!currentTurn.firstAudioLatencyMs) {
        currentTurn.firstAudioLatencyMs =
          Date.now() - (currentTurn.activityEndedAt || currentTurn.startedAt);
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

    if (content.generationComplete || content.turnComplete) {
      handleGenerationDone();
    }
  };

  const connectGemini = (requestedModelIndex = 0) => {
    if (closed) return;

    modelIndex = Math.max(
      0,
      Math.min(requestedModelIndex, models.length - 1),
    );

    closeGemini();
    geminiReady = false;

    const serial = ++connectionSerial;
    const model = models[modelIndex];
    const endpoint =
      "wss://generativelanguage.googleapis.com/ws/" +
      "google.ai.generativelanguage.v1beta.GenerativeService." +
      `BidiGenerateContent?key=${encodeURIComponent(GEMINI_API_KEY)}`;

    console.log(
      `[${deviceId}] Opening Gemini Live model=${model} ` +
        `(attempt ${modelIndex + 1}/${models.length})`,
    );

    const socket = new WebSocket(endpoint, {
      maxPayload: MAX_GEMINI_MESSAGE_BYTES,
      perMessageDeflate: false,
    });
    gemini = socket;

    socket.on("open", () => {
      if (serial !== connectionSerial || closed) return;

      const setup = {
        setup: {
          model: `models/${model}`,
          generationConfig: {
            responseModalities: ["AUDIO"],
            speechConfig: {
              voiceConfig: {
                prebuiltVoiceConfig: {
                  voiceName: ATLAS_VOICE,
                },
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
        },
      };

      socket.send(JSON.stringify(setup));
      console.log(`[${deviceId}] Gemini setup sent.`);

      setupTimer = setTimeout(() => {
        if (serial !== connectionSerial || geminiReady || closed) return;
        console.error(
          `[${deviceId}] Gemini setup timeout model=${model}.`,
        );
        try {
          socket.terminate();
        } catch {}
      }, GEMINI_SETUP_TIMEOUT_MS);
    });

    socket.on("message", (data) => {
      if (serial !== connectionSerial || closed) return;

      try {
        const message = JSON.parse(data.toString());
        handleGeminiMessage(message, serial);
      } catch (error) {
        console.error(
          `[${deviceId}] Gemini message parse failed: ${error.message}`,
        );
      }
    });

    socket.on("error", (error) => {
      if (serial !== connectionSerial || closed) return;
      console.error(
        `[${deviceId}] Gemini socket error model=${model}: ${error.message}`,
      );
    });

    socket.on("close", (code, reasonBuffer) => {
      if (serial !== connectionSerial) return;

      if (setupTimer) clearTimeout(setupTimer);
      setupTimer = null;

      const wasReady = geminiReady;
      geminiReady = false;
      gemini = null;

      const reason = reasonBuffer?.toString() || "";
      console.log(
        `[${deviceId}] Gemini closed: code=${code} model=${model} ` +
          `reason=${reason || "<empty>"}`,
      );

      if (closed) return;

      if (!wasReady && modelIndex + 1 < models.length) {
        connectGemini(modelIndex + 1);
        return;
      }

      sendAtlasJson({
        type: "session_reconnecting",
        reason: "gemini_session_closed",
        code,
      });
      scheduleReconnect(
        GEMINI_RETRY_DELAY_MS,
        "Gemini Live session closed",
      );
    });
  };

  atlas.on("message", (data, isBinary) => {
    if (isBinary) {
      if (
        !geminiReady ||
        !gemini ||
        gemini.readyState !== WebSocket.OPEN ||
        !currentTurn?.active
      ) {
        return;
      }

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

    if (control.type === "ping") {
      sendAtlasJson({ type: "pong", at: Date.now() });
      return;
    }

    if (
      !geminiReady ||
      !gemini ||
      gemini.readyState !== WebSocket.OPEN
    ) {
      sendAtlasJson({
        type: "session_reconnecting",
        reason: "gemini_not_ready",
      });
      return;
    }

    if (control.type === "activity_start") {
      pacer?.clear();
      pacer = null;
      resetTurn();

      gemini.send(
        JSON.stringify({
          realtimeInput: {
            activityStart: {},
          },
        }),
      );

      sendAtlasJson({
        type: "activity_started",
        turnIndex: currentTurn.turnIndex,
      });
      return;
    }

    if (control.type === "activity_end" && currentTurn?.active) {
      currentTurn.activityEndedAt = Date.now();

      gemini.send(
        JSON.stringify({
          realtimeInput: {
            activityEnd: {},
          },
        }),
      );

      sendAtlasJson({
        type: "activity_ended",
        inputBytes: currentTurn.inputBytes,
      });
    }
  });

  atlas.on("error", (error) => {
    console.error(`[${deviceId}] Atlas socket error: ${error.message}`);
  });

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

      // Reliability-first V3.0.2 behavior: discard old raw Live handles.
      if (state?.resume_handle) {
        await clearSessionState(deviceId);
      }

      conversationId =
        state?.conversation_id || (await createConversation(deviceId));

      sendAtlasJson({
        type: "relay_connected",
        version: RELAY_VERSION,
        durableMemory: SUPABASE_ENABLED,
      });

      connectGemini(0);
    } catch (error) {
      console.error(
        `[${deviceId}] relay initialization failed: ${error.message}`,
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
  console.log(`Primary model: ${GEMINI_PRIMARY_MODEL}`);
  console.log(`Fallback model: ${GEMINI_FALLBACK_MODEL}`);
  console.log(`Voice: ${ATLAS_VOICE}`);
  console.log(`Durable memory: ${SUPABASE_ENABLED ? "enabled" : "disabled"}`);
});
