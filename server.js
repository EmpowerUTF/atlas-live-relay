import http from "node:http";
import crypto from "node:crypto";
import { URL } from "node:url";
import { GoogleGenAI, Modality } from "@google/genai";
import { WebSocket, WebSocketServer } from "ws";

const PORT = Number(process.env.PORT || 3000);
const RELAY_VERSION = "3.0.3";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const ATLAS_DEVICE_TOKEN = process.env.ATLAS_DEVICE_TOKEN || "";
const GEMINI_PRIMARY_MODEL =
  process.env.GEMINI_MODEL || "gemini-3.1-flash-live-preview";
const GEMINI_FALLBACK_MODEL =
  process.env.GEMINI_FALLBACK_MODEL ||
  "gemini-2.5-flash-native-audio-preview-12-2025";
const GEMINI_RETRY_DELAY_MS = 1500;
const ATLAS_VOICE = process.env.ATLAS_VOICE || "Kore";
const BASE_SYSTEM_INSTRUCTION =
  process.env.ATLAS_SYSTEM_INSTRUCTION ||
  "You are Atlas, a fast, practical AI companion. Speak naturally and directly. " +
    "Keep routine replies concise. Ask one useful follow-up only when it is genuinely needed.";

const SUPABASE_URL = (process.env.SUPABASE_URL || "").replace(/\/$/, "");
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || "";
const SUPABASE_ENABLED = Boolean(SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY);

const MAX_ATLAS_MESSAGE_BYTES = 64 * 1024;
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
  let instruction = BASE_SYSTEM_INSTRUCTION;

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
      model: GEMINI_PRIMARY_MODEL,
      fallbackModel: GEMINI_FALLBACK_MODEL,
      voice: ATLAS_VOICE,
      mode: "persistent-websocket-official-genai-sdk",
      geminiTransport: "official-google-genai-sdk",
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

  let geminiSession = null;
  let geminiReady = false;
  let geminiConnecting = false;
  let closed = false;

  let conversationId = null;
  let recentTurns = [];
  let memories = [];

  let turnCounter = 0;
  let currentTurn = null;
  let pacer = null;

  let activeGeminiModel = GEMINI_PRIMARY_MODEL;
  let activeModelIndex = 0;
  let geminiConnectionSerial = 0;
  let geminiRetryTimer = null;

  const geminiModels = [...new Set(
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
      // Input/output transcription events can arrive slightly after audio
      // generation completes. Keep the latest fragments before saving.
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

      void saveTurn(savedTurn).catch((error) =>
        console.error("Save turn failed:", error.message),
      );
    }, TRANSCRIPT_SETTLE_MS);
  };

  const closeGeminiSession = () => {
    const session = geminiSession;
    geminiSession = null;
    geminiReady = false;
    geminiConnecting = false;

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

  const closeAll = (code = 1000, reason = "session_closed") => {
    if (closed) return;
    closed = true;

    if (geminiRetryTimer) clearTimeout(geminiRetryTimer);
    geminiRetryTimer = null;

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

  const handleGeminiMessage = (message, connectionSerial) => {
    if (
      closed ||
      connectionSerial !== geminiConnectionSerial ||
      !message
    ) {
      return;
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
        if (!currentTurn) resetTurn();

        if (!currentTurn.firstAudioLatencyMs) {
          currentTurn.firstAudioLatencyMs =
            Date.now() - currentTurn.startedAt;
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

      if (content.generationComplete) {
        handleGenerationDone();
      } else if (
        content.turnComplete &&
        !pacer &&
        currentTurn?.active
      ) {
        sendAtlasJson({ type: "turn_complete" });
        finishTurn();
      }
    }

    if (message.usageMetadata && currentTurn) {
      currentTurn.usage = message.usageMetadata;
    }

    if (message.goAway) {
      console.log(
        `[${deviceId}] Gemini requested session rotation ` +
          `model=${activeGeminiModel}.`,
      );
      sendAtlasJson({ type: "go_away", detail: message.goAway });

      setTimeout(() => {
        if (
          !closed &&
          connectionSerial === geminiConnectionSerial
        ) {
          closeGeminiSession();
          scheduleGeminiConnect(0, 500, "Gemini session rotation");
        }
      }, 250);
    }
  };

  const scheduleGeminiConnect = (modelIndex, delayMs, reason) => {
    if (closed) return;

    if (geminiRetryTimer) clearTimeout(geminiRetryTimer);

    if (reason) {
      console.log(
        `[${deviceId}] Gemini reconnect scheduled in ${delayMs} ms: ${reason}`,
      );
    }

    geminiRetryTimer = setTimeout(() => {
      geminiRetryTimer = null;
      void connectGemini(modelIndex);
    }, delayMs);
  };

  const connectGemini = async (requestedModelIndex = 0) => {
    if (closed || geminiConnecting) return;

    activeModelIndex = Math.max(
      0,
      Math.min(requestedModelIndex, geminiModels.length - 1),
    );
    activeGeminiModel = geminiModels[activeModelIndex];

    const connectionSerial = ++geminiConnectionSerial;
    geminiConnecting = true;
    geminiReady = false;

    closeGeminiSession();
    geminiConnecting = true;

    console.log(
      `[${deviceId}] Opening Gemini Live through official SDK ` +
        `model=${activeGeminiModel} ` +
        `(attempt ${activeModelIndex + 1}/${geminiModels.length})`,
    );

    const config = {
      responseModalities: [Modality.AUDIO],
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
              `[${deviceId}] Gemini SDK socket opened ` +
                `model=${activeGeminiModel}.`,
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
              `[${deviceId}] Gemini SDK error ` +
                `model=${activeGeminiModel}: ${detail}`,
            );
          },

          onclose: (event) => {
            if (connectionSerial !== geminiConnectionSerial) return;

            const wasReady = geminiReady;
            geminiReady = false;
            geminiConnecting = false;
            geminiSession = null;

            const code = event?.code ?? 0;
            const reason = event?.reason || "<empty>";

            console.log(
              `[${deviceId}] Gemini SDK closed: code=${code} ` +
                `model=${activeGeminiModel} reason=${reason}`,
            );

            if (closed) return;

            // If the socket closes before live.connect() resolves, its
            // rejected promise handles model fallback. Only reconnect here
            // when a previously established session later ends.
            if (!wasReady) return;

            scheduleGeminiConnect(
              0,
              GEMINI_RETRY_DELAY_MS,
              "established Live session closed",
            );
          },
        },
      });

      if (
        closed ||
        connectionSerial !== geminiConnectionSerial
      ) {
        try {
          session.close();
        } catch {}
        return;
      }

      // The SDK connect promise resolves only after the Live session setup
      // has completed. This replaces our faulty hand-built setup handshake.
      geminiSession = session;
      geminiConnecting = false;
      geminiReady = true;

      sendAtlasJson({
        type: "ready",
        model: activeGeminiModel,
        voice: ATLAS_VOICE,
        conversationId,
        resumed: false,
        durableMemory: SUPABASE_ENABLED,
        memoryCount: memories.length,
      });

      console.log(
        `[${deviceId}] Gemini Live ready through official SDK. ` +
          `model=${activeGeminiModel} ` +
          `savedContextTurns=${recentTurns.length}`,
      );
    } catch (error) {
      if (connectionSerial !== geminiConnectionSerial || closed) return;

      geminiSession = null;
      geminiReady = false;
      geminiConnecting = false;

      console.error(
        `[${deviceId}] Gemini SDK connect failed ` +
          `model=${activeGeminiModel}: ${error.message}`,
      );

      const nextModelIndex = activeModelIndex + 1;
      if (nextModelIndex < geminiModels.length) {
        scheduleGeminiConnect(
          nextModelIndex,
          350,
          `primary SDK connection failed: ${error.message}`,
        );
        return;
      }

      sendAtlasJson({
        type: "error",
        source: "gemini",
        message:
          "Gemini Live setup failed through the official SDK on all " +
          `configured models. Last error: ${error.message}`,
      });

      scheduleGeminiConnect(
        0,
        5000,
        "all official SDK model connections failed",
      );
    }
  };

  atlas.on("message", (data, isBinary) => {
    if (!geminiReady || !geminiSession) return;

    if (isBinary) {
      if (!currentTurn?.active) return;

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
        console.error(
          `[${deviceId}] Gemini audio send failed: ${error.message}`,
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

    if (control.type === "activity_start") {
      pacer?.clear();
      pacer = null;
      resetTurn();

      try {
        geminiSession.sendRealtimeInput({ activityStart: {} });
        sendAtlasJson({
          type: "activity_started",
          turnIndex: currentTurn.turnIndex,
        });
      } catch (error) {
        console.error(
          `[${deviceId}] activityStart failed: ${error.message}`,
        );
      }
    } else if (control.type === "activity_end") {
      if (currentTurn?.active) {
        try {
          geminiSession.sendRealtimeInput({ activityEnd: {} });
          sendAtlasJson({
            type: "activity_ended",
            inputBytes: currentTurn.inputBytes,
          });
        } catch (error) {
          console.error(
            `[${deviceId}] activityEnd failed: ${error.message}`,
          );
        }
      }
    } else if (control.type === "ping") {
      sendAtlasJson({ type: "pong", at: Date.now() });
    }
  });

  atlas.on("error", (error) => {
    console.error(`[${deviceId}] Atlas socket error:`, error.message);
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

      // Old raw-WebSocket session handles are deliberately cleared. Durable
      // continuity is supplied from saved transcripts and memories instead.
      if (state?.resume_handle) {
        await clearSessionState(deviceId);
      }

      conversationId =
        state?.conversation_id || (await createConversation(deviceId));

      sendAtlasJson({
        type: "relay_connected",
        version: RELAY_VERSION,
        durableMemory: SUPABASE_ENABLED,
        geminiTransport: "official-google-genai-sdk",
      });

      await connectGemini(0);
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
  console.log("Gemini transport: official @google/genai Live SDK");
  console.log(`Primary model: ${GEMINI_PRIMARY_MODEL}`);
  console.log(`Fallback model: ${GEMINI_FALLBACK_MODEL}`);
  console.log(`Voice: ${ATLAS_VOICE}`);
  console.log(`Durable memory: ${SUPABASE_ENABLED ? "enabled" : "disabled"}`);
});
