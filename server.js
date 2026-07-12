import http from "node:http";
import crypto from "node:crypto";
import { URL } from "node:url";
import { GoogleGenAI, Modality } from "@google/genai";

const PORT = Number(process.env.PORT || 3000);
const RELAY_VERSION = "5.0.0-one-press-http-persistent-live";

const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const ATLAS_DEVICE_TOKEN = process.env.ATLAS_DEVICE_TOKEN || "";
const GEMINI_PRIMARY_MODEL =
  process.env.GEMINI_MODEL || "gemini-3.1-flash-live-preview";
const GEMINI_FALLBACK_MODEL =
  process.env.GEMINI_FALLBACK_MODEL ||
  "gemini-2.5-flash-native-audio-preview-12-2025";
const ATLAS_VOICE = process.env.ATLAS_VOICE || "Charon";

const BASE_SYSTEM_INSTRUCTION =
  process.env.ATLAS_SYSTEM_INSTRUCTION ||
  "You are Atlas, a fast, practical AI companion. Answer the owner's spoken request directly. " +
    "Do not begin with a greeting unless the owner greeted you. Keep routine answers concise, " +
    "but give enough detail to be useful. If the speech is genuinely unintelligible, say exactly: " +
    "I didn't catch that. Could you say it again?";

const VOICE_AND_LANGUAGE_LOCK =
  "Always respond in English unless the owner explicitly asks to switch language. " +
  "Use an articulate adult British male delivery: cultivated modern Received Pronunciation, " +
  "measured, warm, intelligent, natural, and not theatrical. Do not copy an accent inferred " +
  "from noisy audio. Avoid exaggerated aristocratic affectation and avoid rushing.";

const SUPABASE_URL = (process.env.SUPABASE_URL || "").replace(/\/$/, "");
const SUPABASE_SERVICE_ROLE_KEY =
  process.env.SUPABASE_SERVICE_ROLE_KEY || "";
const SUPABASE_ENABLED = Boolean(
  SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY,
);

const MAX_INPUT_BYTES = 320_000; // 10 seconds of 16 kHz mono PCM16.
const MAX_OUTPUT_BYTES = 768_000; // 16 seconds of 24 kHz mono PCM16.
const GEMINI_TURN_TIMEOUT_MS = 35_000;
const TRANSCRIPT_SETTLE_MS = 300;
const SESSION_ROTATE_MS = 8 * 60 * 1000;
const RECONNECT_DELAY_MS = 1000;
const RECENT_HISTORY_TURNS = 8;
const INPUT_CHUNK_BYTES = 16_000; // 500 ms per Live API audio message.

if (!GEMINI_API_KEY) {
  console.error("Missing required environment variable: GEMINI_API_KEY");
  process.exit(1);
}
if (!ATLAS_DEVICE_TOKEN) {
  console.error("Missing required environment variable: ATLAS_DEVICE_TOKEN");
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

let liveSession = null;
let liveModel = "";
let liveOpenedAt = 0;
let liveSerial = 0;
let connectPromise = null;
let reconnectTimer = null;
let rotateAfterTurn = false;
let activeTurn = null;
let turnRequestInProgress = false;
let turnIndex = 0;
let conversationId = crypto.randomUUID();
let recentTurns = [];
let memories = [];
let contextLoaded = false;
let lastTurnDebug = null;

function jsonResponse(res, status, payload) {
  if (res.writableEnded) return;
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": String(Buffer.byteLength(body)),
    "cache-control": "no-store",
    connection: "close",
  });
  res.end(body);
}

function authorized(parsed) {
  return parsed.searchParams.get("token") === ATLAS_DEVICE_TOKEN;
}

function safeDeviceId(value) {
  const cleaned = String(value || "atlas-v1").replace(
    /[^a-zA-Z0-9_.-]/g,
    "",
  );
  return cleaned.slice(0, 64) || "atlas-v1";
}

function mergeTranscript(current, fragment) {
  const next = String(fragment || "").trim();
  if (!next) return String(current || "").trim();

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

async function createConversation(deviceId) {
  conversationId = crypto.randomUUID();
  if (!SUPABASE_ENABLED) return;

  await supabaseRequest("/rest/v1/atlas_conversations", {
    method: "POST",
    prefer: "return=minimal",
    body: {
      id: conversationId,
      device_id: deviceId,
      started_at: new Date().toISOString(),
      last_active_at: new Date().toISOString(),
    },
  });
}

async function saveTurn(turn) {
  if (!SUPABASE_ENABLED) return;

  await supabaseRequest("/rest/v1/atlas_turns", {
    method: "POST",
    prefer: "return=minimal",
    body: {
      conversation_id: conversationId,
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

  await supabaseRequest(
    `/rest/v1/atlas_conversations?id=eq.${encodeURIComponent(conversationId)}`,
    {
      method: "PATCH",
      prefer: "return=minimal",
      body: { last_active_at: new Date().toISOString() },
    },
  );
}

async function loadContextOnce(deviceId = "atlas-v1") {
  if (contextLoaded) return;

  try {
    const [storedTurns, storedMemories] = await Promise.all([
      loadRecentTurns(deviceId),
      loadMemories(deviceId),
    ]);
    recentTurns = storedTurns;
    memories = storedMemories;
    await createConversation(deviceId);
  } catch (error) {
    console.error("Atlas storage context load failed:", error.message);
    // Voice must remain usable even if Supabase is temporarily unavailable.
    conversationId = crypto.randomUUID();
  }

  contextLoaded = true;
}

function buildSystemInstruction() {
  let instruction = `${BASE_SYSTEM_INSTRUCTION}\n\n${VOICE_AND_LANGUAGE_LOCK}`;

  if (memories.length) {
    instruction +=
      "\n\nDurable owner memory. Use only when relevant and never mention the storage system:\n" +
      memories.map((item) => `- ${item.content}`).join("\n");
  }

  if (recentTurns.length) {
    const lines = [];
    for (const turn of recentTurns.slice(-RECENT_HISTORY_TURNS)) {
      if (turn.user_text) lines.push(`Owner: ${turn.user_text}`);
      if (turn.atlas_text) lines.push(`Atlas: ${turn.atlas_text}`);
    }
    if (lines.length) {
      instruction +=
        "\n\nRecent conversation context. Continue naturally when relevant:\n" +
        lines.join("\n");
    }
  }

  return instruction;
}

function closeLiveSession(reason = "rotation") {
  const session = liveSession;
  liveSession = null;
  liveModel = "";
  liveOpenedAt = 0;
  liveSerial += 1;

  if (session) {
    try {
      session.close();
    } catch (error) {
      console.error(`Gemini close error (${reason}):`, error.message);
    }
  }
}

function scheduleReconnect(reason) {
  if (reconnectTimer || activeTurn) return;

  console.log(`Gemini reconnect scheduled: ${reason}`);
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    void ensureLiveSession(true).catch((error) => {
      console.error("Gemini background reconnect failed:", error.message);
      scheduleReconnect("retry after failed reconnect");
    });
  }, RECONNECT_DELAY_MS);
}

function failActiveTurn(status, message) {
  const turn = activeTurn;
  if (!turn || turn.finished) return;

  turn.finished = true;
  if (turn.timeout) clearTimeout(turn.timeout);
  if (turn.finishTimer) clearTimeout(turn.finishTimer);

  console.error(`[${turn.deviceId}] turn failed: ${message}`);

  if (!turn.res.writableEnded) {
    if (!turn.res.headersSent) {
      jsonResponse(turn.res, status, { ok: false, error: message });
    } else {
      turn.res.destroy();
    }
  }

  activeTurn = null;
  closeLiveSession(`turn failure: ${message}`);
  scheduleReconnect("turn failure recovery");
}

function finishActiveTurn(reason) {
  const turn = activeTurn;
  if (!turn || turn.finished) return;

  turn.finished = true;
  if (turn.timeout) clearTimeout(turn.timeout);
  if (turn.finishTimer) clearTimeout(turn.finishTimer);

  if (turn.outputBytes === 0) {
    activeTurn = null;
    if (!turn.res.headersSent) {
      jsonResponse(turn.res, 502, {
        ok: false,
        error: "gemini_returned_no_audio",
      });
    } else if (!turn.res.writableEnded) {
      turn.res.end();
    }
    closeLiveSession("no output audio");
    scheduleReconnect("no output recovery");
    return;
  }

  if (!turn.res.writableEnded) turn.res.end();

  turn.totalTurnMs = Date.now() - turn.startedAt;
  lastTurnDebug = {
    createdAt: new Date().toISOString(),
    deviceId: turn.deviceId,
    inputBytes: turn.inputBytes,
    outputBytes: turn.outputBytes,
    inputTranscript: turn.inputTranscript,
    outputTranscript: turn.outputTranscript,
    firstAudioLatencyMs: turn.firstAudioLatencyMs,
    totalTurnMs: turn.totalTurnMs,
    reason,
    model: liveModel,
  };

  console.log(
    `[${turn.deviceId}] response complete (${reason}). ` +
      `input=${turn.inputBytes} output=${turn.outputBytes} ` +
      `firstAudio=${turn.firstAudioLatencyMs || 0}ms total=${turn.totalTurnMs}ms`,
  );
  console.log(
    `[${turn.deviceId}] INPUT TRANSCRIPT: ${turn.inputTranscript || "<empty>"}`,
  );
  console.log(
    `[${turn.deviceId}] OUTPUT TRANSCRIPT: ${turn.outputTranscript || "<empty>"}`,
  );

  recentTurns.push({
    user_text: turn.inputTranscript,
    atlas_text: turn.outputTranscript,
    created_at: new Date().toISOString(),
  });
  if (recentTurns.length > RECENT_HISTORY_TURNS) {
    recentTurns = recentTurns.slice(-RECENT_HISTORY_TURNS);
  }

  activeTurn = null;

  void saveTurn(turn).catch((error) => {
    console.error("Supabase turn save failed:", error.message);
  });

  if (rotateAfterTurn || Date.now() - liveOpenedAt >= SESSION_ROTATE_MS) {
    rotateAfterTurn = false;
    closeLiveSession("scheduled rotation after turn");
    scheduleReconnect("scheduled session rotation");
  }
}

function scheduleTurnFinish(reason) {
  const turn = activeTurn;
  if (!turn || turn.finished || turn.finishTimer) return;

  turn.finishTimer = setTimeout(() => {
    if (activeTurn === turn) finishActiveTurn(reason);
  }, TRANSCRIPT_SETTLE_MS);
}

function handleGeminiMessage(message, serial) {
  if (serial !== liveSerial) return;

  if (message?.usageMetadata && activeTurn) {
    activeTurn.usage = message.usageMetadata;
  }

  if (message?.sessionResumptionUpdate?.newHandle) {
    // We intentionally do not restore raw Live handles. Rebuilding from saved
    // transcripts proved more reliable during earlier Atlas testing.
  }

  if (message?.goAway) {
    console.log("Gemini sent GoAway; session will rotate cleanly.");
    if (activeTurn) rotateAfterTurn = true;
    else {
      closeLiveSession("Gemini GoAway");
      scheduleReconnect("Gemini GoAway");
    }
  }

  const content = message?.serverContent;
  if (!content || !activeTurn) return;

  if (content.inputTranscription?.text) {
    activeTurn.inputTranscript = mergeTranscript(
      activeTurn.inputTranscript,
      content.inputTranscription.text,
    );
  }

  if (content.outputTranscription?.text) {
    activeTurn.outputTranscript = mergeTranscript(
      activeTurn.outputTranscript,
      content.outputTranscription.text,
    );
  }

  if (content.interrupted) {
    failActiveTurn(409, "gemini_interrupted");
    return;
  }

  const parts = content.modelTurn?.parts || [];
  for (const part of parts) {
    const inline = part.inlineData;
    if (!inline?.data || !(inline.mimeType || "").startsWith("audio/pcm")) {
      continue;
    }

    const pcm = Buffer.from(inline.data, "base64");
    if (!pcm.length) continue;

    if (activeTurn.outputBytes + pcm.length > MAX_OUTPUT_BYTES) {
      failActiveTurn(502, "gemini_audio_too_long");
      return;
    }

    if (!activeTurn.res.headersSent) {
      activeTurn.firstAudioLatencyMs =
        Date.now() - activeTurn.activityEndedAt;
      activeTurn.res.writeHead(200, {
        "content-type": "audio/pcm;rate=24000",
        "cache-control": "no-store",
        connection: "close",
        "x-atlas-relay-version": RELAY_VERSION,
        "x-atlas-model": liveModel,
        "x-atlas-voice": ATLAS_VOICE,
      });
      console.log(
        `[${activeTurn.deviceId}] first Gemini audio after ` +
          `${activeTurn.firstAudioLatencyMs} ms.`,
      );
    }

    activeTurn.outputBytes += pcm.length;
    activeTurn.res.write(pcm);
  }

  if (content.generationComplete || content.turnComplete) {
    scheduleTurnFinish(
      content.generationComplete ? "generationComplete" : "turnComplete",
    );
  }
}

async function connectModel(model) {
  const serial = ++liveSerial;
  const config = {
    responseModalities: [Modality.AUDIO],
    systemInstruction: buildSystemInstruction(),
    inputAudioTranscription: {},
    outputAudioTranscription: {},
    realtimeInputConfig: {
      automaticActivityDetection: { disabled: true },
    },
    speechConfig: {
      voiceConfig: {
        prebuiltVoiceConfig: { voiceName: ATLAS_VOICE },
      },
    },
  };

  if (model.includes("3.1")) {
    config.thinkingConfig = { thinkingLevel: "minimal" };
  }

  console.log(`Opening persistent Gemini Live session: ${model}`);

  const session = await ai.live.connect({
    model,
    config,
    callbacks: {
      onopen: () => {
        if (serial === liveSerial) {
          console.log(`Gemini Live WebSocket opened: ${model}`);
        }
      },
      onmessage: (message) => handleGeminiMessage(message, serial),
      onerror: (event) => {
        if (serial !== liveSerial) return;
        const detail = event?.message || String(event);
        console.error(`Gemini Live error (${model}): ${detail}`);
        if (activeTurn) failActiveTurn(502, `gemini_error: ${detail}`);
      },
      onclose: (event) => {
        if (serial !== liveSerial) return;
        const code = event?.code ?? 1000;
        const reason = event?.reason || "";
        console.log(`Gemini Live closed: ${code} ${reason}`);
        liveSession = null;
        liveModel = "";
        liveOpenedAt = 0;
        if (activeTurn) {
          failActiveTurn(503, `gemini_session_closed_${code}`);
        } else {
          scheduleReconnect("Gemini socket closed");
        }
      },
    },
  });

  if (serial !== liveSerial) {
    try {
      session.close();
    } catch {}
    throw new Error("superseded Gemini connection");
  }

  liveSession = session;
  liveModel = model;
  liveOpenedAt = Date.now();
  console.log(`Persistent Gemini Live session ready: ${model}`);
  return session;
}

async function ensureLiveSession(forceReconnect = false) {
  await loadContextOnce("atlas-v1");

  const stale =
    liveSession && Date.now() - liveOpenedAt >= SESSION_ROTATE_MS;

  if (liveSession && !forceReconnect && !stale) return liveSession;
  if (connectPromise) return connectPromise;

  if (liveSession && (forceReconnect || stale)) {
    closeLiveSession(forceReconnect ? "forced reconnect" : "age rotation");
  }

  connectPromise = (async () => {
    const models = [GEMINI_PRIMARY_MODEL, GEMINI_FALLBACK_MODEL].filter(
      (model, index, array) => model && array.indexOf(model) === index,
    );

    let lastError = null;
    for (const model of models) {
      try {
        return await connectModel(model);
      } catch (error) {
        lastError = error;
        console.error(`Gemini model connection failed (${model}):`, error.message);
      }
    }

    throw lastError || new Error("no Gemini Live model available");
  })();

  try {
    return await connectPromise;
  } finally {
    connectPromise = null;
  }
}

function collectPcmBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let total = 0;
    let settled = false;

    const fail = (error) => {
      if (settled) return;
      settled = true;
      reject(error);
    };

    req.on("data", (chunk) => {
      if (settled) return;
      total += chunk.length;
      if (total > MAX_INPUT_BYTES) {
        fail(Object.assign(new Error("input_too_large"), { status: 413 }));
        req.destroy();
        return;
      }
      chunks.push(Buffer.from(chunk));
    });

    req.on("end", () => {
      if (settled) return;
      settled = true;
      resolve(Buffer.concat(chunks, total));
    });

    req.on("aborted", () => {
      fail(Object.assign(new Error("client_aborted_upload"), { status: 499 }));
    });

    req.on("error", (error) => {
      fail(
        Object.assign(new Error(`request_error: ${error.message}`), {
          status: 400,
        }),
      );
    });
  });
}

function submitTurnToGemini(audio, res, deviceId) {
  if (!liveSession) throw new Error("Gemini session is not ready");
  if (activeTurn) {
    throw Object.assign(new Error("atlas_busy"), { status: 409 });
  }

  const turn = {
    deviceId,
    turnIndex: ++turnIndex,
    startedAt: Date.now(),
    activityEndedAt: 0,
    inputBytes: audio.length,
    outputBytes: 0,
    inputTranscript: "",
    outputTranscript: "",
    firstAudioLatencyMs: 0,
    totalTurnMs: 0,
    usage: {},
    res,
    timeout: null,
    finishTimer: null,
    finished: false,
  };

  activeTurn = turn;

  res.on("close", () => {
    if (activeTurn === turn && !turn.finished && !res.writableEnded) {
      failActiveTurn(499, "client_disconnected_during_response");
    }
  });

  try {
    liveSession.sendRealtimeInput({ activityStart: {} });

    for (let offset = 0; offset < audio.length; offset += INPUT_CHUNK_BYTES) {
      const chunk = audio.subarray(
        offset,
        Math.min(offset + INPUT_CHUNK_BYTES, audio.length),
      );
      liveSession.sendRealtimeInput({
        audio: {
          data: chunk.toString("base64"),
          mimeType: "audio/pcm;rate=16000",
        },
      });
    }

    liveSession.sendRealtimeInput({ activityEnd: {} });
    turn.activityEndedAt = Date.now();

    console.log(
      `[${deviceId}] PCM submitted to persistent Gemini session. ` +
        `bytes=${audio.length} model=${liveModel}`,
    );

    turn.timeout = setTimeout(() => {
      if (activeTurn === turn) {
        failActiveTurn(504, "gemini_reply_timeout");
      }
    }, GEMINI_TURN_TIMEOUT_MS);
  } catch (error) {
    failActiveTurn(502, `gemini_submit_failed: ${error.message}`);
  }
}

async function handleTurn(req, res, parsed) {
  if (!authorized(parsed)) {
    jsonResponse(res, 401, { ok: false, error: "unauthorized" });
    return;
  }

  if (activeTurn || turnRequestInProgress) {
    jsonResponse(res, 409, { ok: false, error: "atlas_busy" });
    return;
  }

  turnRequestInProgress = true;
  const deviceId = safeDeviceId(parsed.searchParams.get("device"));
  console.log(`[${deviceId}] HTTP one-press turn request received.`);

  try {
    // Warm/reconnect Gemini while the ESP32 is still uploading its PCM body.
    const sessionPromise = ensureLiveSession(false);
    const audio = await collectPcmBody(req);
    await sessionPromise;

    if (!audio.length) {
      turnRequestInProgress = false;
      jsonResponse(res, 400, { ok: false, error: "empty_audio" });
      return;
    }
    if (audio.length % 2 !== 0) {
      turnRequestInProgress = false;
      jsonResponse(res, 400, { ok: false, error: "pcm_length_must_be_even" });
      return;
    }

    console.log(`[${deviceId}] HTTP upload complete. Bytes: ${audio.length}`);
    submitTurnToGemini(audio, res, deviceId);
    turnRequestInProgress = false;
  } catch (error) {
    turnRequestInProgress = false;
    const status = error?.status || 503;
    const message = error?.message || String(error);
    console.error(`[${deviceId}] HTTP turn failed: ${message}`);
    if (!res.headersSent) {
      jsonResponse(res, status, { ok: false, error: message });
    } else if (!res.writableEnded) {
      res.destroy();
    }
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
      mode: "one-press-local-vad-http-to-persistent-gemini-live",
      geminiReady: Boolean(liveSession),
      geminiConnecting: Boolean(connectPromise),
      activeModel: liveModel || null,
      primaryModel: GEMINI_PRIMARY_MODEL,
      fallbackModel: GEMINI_FALLBACK_MODEL,
      voice: ATLAS_VOICE,
      durableMemory: SUPABASE_ENABLED,
      busy: Boolean(activeTurn),
    });
    return;
  }

  if (req.method === "GET" && parsed.pathname === "/warm") {
    if (!authorized(parsed)) {
      jsonResponse(res, 401, { ok: false, error: "unauthorized" });
      return;
    }

    try {
      await ensureLiveSession(false);
      jsonResponse(res, 200, {
        ok: true,
        version: RELAY_VERSION,
        geminiReady: true,
        model: liveModel,
      });
    } catch (error) {
      jsonResponse(res, 503, { ok: false, error: error.message });
    }
    return;
  }

  if (req.method === "POST" && parsed.pathname === "/turn") {
    await handleTurn(req, res, parsed);
    return;
  }

  if (req.method === "GET" && parsed.pathname === "/debug/last-turn") {
    if (!authorized(parsed)) {
      jsonResponse(res, 401, { ok: false, error: "unauthorized" });
      return;
    }
    jsonResponse(res, 200, {
      ok: true,
      turn: lastTurnDebug,
    });
    return;
  }

  if (req.method === "GET" && parsed.pathname === "/api/chats") {
    if (!authorized(parsed)) {
      jsonResponse(res, 401, { ok: false, error: "unauthorized" });
      return;
    }

    const deviceId = safeDeviceId(parsed.searchParams.get("device"));
    try {
      const turns = await loadRecentTurns(deviceId);
      jsonResponse(res, 200, { ok: true, deviceId, turns });
    } catch (error) {
      jsonResponse(res, 503, { ok: false, error: error.message });
    }
    return;
  }

  jsonResponse(res, 404, { ok: false, error: "not_found" });
});

server.keepAliveTimeout = 65_000;
server.headersTimeout = 70_000;
server.requestTimeout = 0;

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Atlas Relay ${RELAY_VERSION} listening on port ${PORT}`);
  console.log("ESP32 transport: one HTTPS POST per completed local-VAD turn");
  console.log("Gemini transport: persistent official @google/genai Live session");
  console.log(`Primary model: ${GEMINI_PRIMARY_MODEL}`);
  console.log(`Fallback model: ${GEMINI_FALLBACK_MODEL}`);
  console.log(`Voice: ${ATLAS_VOICE}`);
  console.log(`Durable memory: ${SUPABASE_ENABLED ? "enabled" : "disabled"}`);

  // Prewarm Gemini when Railway starts. A failed prewarm does not crash the
  // relay; /turn will retry while the ESP32 uploads its audio.
  void ensureLiveSession(false).catch((error) => {
    console.error("Initial Gemini prewarm failed:", error.message);
    scheduleReconnect("initial prewarm failure");
  });
});
