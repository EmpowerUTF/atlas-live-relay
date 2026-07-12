import http from "node:http";
import { URL } from "node:url";

const PORT = Number(process.env.PORT || 3000);
const VERSION = "3.8.4-http-single-method";
const API_KEY = process.env.GEMINI_API_KEY || "";
const DEVICE_TOKEN = process.env.ATLAS_DEVICE_TOKEN || "";
const MODEL = process.env.GEMINI_MODEL || "gemini-3.1-flash-live-preview";
const FALLBACK_MODEL =
  process.env.GEMINI_FALLBACK_MODEL ||
  "gemini-2.5-flash-native-audio-preview-12-2025";
const VOICE = process.env.ATLAS_VOICE || "Orus";
const SYSTEM_INSTRUCTION =
  process.env.ATLAS_SYSTEM_INSTRUCTION ||
  "You are Atlas, a fast, practical AI companion. Answer the owner's spoken request directly. " +
    "Do not greet unless greeted. Keep ordinary answers concise. If speech is genuinely unclear, " +
    "say: I didn't catch that. Could you say it again?";

const MAX_INPUT_BYTES = 320_000;
const MAX_OUTPUT_BYTES = 768_000;
const INPUT_CHUNK_BYTES = 16_000;
const TURN_TIMEOUT_MS = 35_000;
const FINISH_SETTLE_MS = 250;
const RECONNECT_DELAY_MS = 1_000;

let GoogleGenAI;
let Modality;
let ai;
let session = null;
let sessionModel = "";
let sessionSerial = 0;
let connectPromise = null;
let reconnectTimer = null;
let activeTurn = null;
let requestInProgress = false;
let completedTurns = 0;
let sessionConnections = 0;

function json(res, status, body) {
  if (res.writableEnded) return;
  const data = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": String(Buffer.byteLength(data)),
    "cache-control": "no-store",
    connection: "close",
  });
  res.end(data);
}

function authorized(parsed) {
  return DEVICE_TOKEN && parsed.searchParams.get("token") === DEVICE_TOKEN;
}

async function loadSdk() {
  if (GoogleGenAI) return;

  if (process.env.ATLAS_MOCK_GEMINI === "1") {
    Modality = { AUDIO: "AUDIO" };
    GoogleGenAI = class MockGoogleGenAI {
      constructor() {
        this.live = {
          connect: async ({ callbacks }) => {
            sessionConnections += 1;
            queueMicrotask(() => callbacks.onopen?.());
            return {
              sendRealtimeInput(message) {
                if (message.activityEnd) {
                  setTimeout(() => {
                    const pcm = Buffer.alloc(9_600, 0x01).toString("base64");
                    callbacks.onmessage?.({
                      serverContent: {
                        modelTurn: {
                          parts: [
                            { inlineData: { data: pcm, mimeType: "audio/pcm;rate=24000" } },
                          ],
                        },
                      },
                    });
                    callbacks.onmessage?.({
                      serverContent: { turnComplete: true },
                    });
                  }, 10);
                }
              },
              close() {
                callbacks.onclose?.({ code: 1000, reason: "mock close" });
              },
            };
          },
        };
      }
    };
    return;
  }

  const sdk = await import("@google/genai");
  GoogleGenAI = sdk.GoogleGenAI;
  Modality = sdk.Modality;
}

function clearSession(reason) {
  const old = session;
  session = null;
  sessionModel = "";
  sessionSerial += 1;
  if (old) {
    try {
      old.close();
    } catch (error) {
      console.error(`Gemini close error (${reason}):`, error.message);
    }
  }
}

function scheduleReconnect(reason) {
  if (reconnectTimer || activeTurn || !API_KEY) return;
  console.log(`Gemini reconnect scheduled: ${reason}`);
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    void ensureSession(true).catch((error) => {
      console.error("Gemini reconnect failed:", error.message);
      scheduleReconnect("retry");
    });
  }, RECONNECT_DELAY_MS);
}

function failTurn(status, message, resetSession = true) {
  const turn = activeTurn;
  if (!turn || turn.finished) return;
  turn.finished = true;
  clearTimeout(turn.timeout);
  clearTimeout(turn.finishTimer);
  console.error(`Turn failed: ${message}`);

  if (!turn.res.writableEnded) {
    if (!turn.res.headersSent) json(turn.res, status, { ok: false, error: message });
    else turn.res.destroy();
  }

  activeTurn = null;
  if (resetSession) {
    clearSession(message);
    scheduleReconnect("turn recovery");
  }
}

function finishTurn(reason) {
  const turn = activeTurn;
  if (!turn || turn.finished) return;
  turn.finished = true;
  clearTimeout(turn.timeout);
  clearTimeout(turn.finishTimer);

  if (!turn.outputBytes) {
    activeTurn = null;
    if (!turn.res.headersSent) {
      json(turn.res, 502, { ok: false, error: "gemini_returned_no_audio" });
    } else if (!turn.res.writableEnded) {
      turn.res.end();
    }
    clearSession("no audio");
    scheduleReconnect("no audio recovery");
    return;
  }

  if (!turn.res.writableEnded) turn.res.end();
  completedTurns += 1;
  console.log(
    `Atlas HTTP response complete (${reason}). input=${turn.inputBytes} ` +
      `output=${turn.outputBytes} total=${Date.now() - turn.startedAt}ms ` +
      `session=${sessionSerial}`,
  );
  activeTurn = null;
}

function scheduleFinish(reason) {
  const turn = activeTurn;
  if (!turn || turn.finished || turn.finishTimer) return;
  turn.finishTimer = setTimeout(() => {
    if (activeTurn === turn) finishTurn(reason);
  }, FINISH_SETTLE_MS);
}

function handleGeminiMessage(message, serial) {
  if (serial !== sessionSerial) return;

  if (message?.goAway) {
    console.log("Gemini GoAway received; reconnecting after this turn.");
    if (!activeTurn) {
      clearSession("GoAway");
      scheduleReconnect("GoAway");
    }
  }

  const content = message?.serverContent;
  if (!content || !activeTurn) return;

  if (content.interrupted) {
    failTurn(409, "gemini_interrupted");
    return;
  }

  for (const part of content.modelTurn?.parts || []) {
    const inline = part.inlineData;
    if (!inline?.data || !(inline.mimeType || "").startsWith("audio/pcm")) continue;

    const pcm = Buffer.from(inline.data, "base64");
    if (!pcm.length) continue;
    if (activeTurn.outputBytes + pcm.length > MAX_OUTPUT_BYTES) {
      failTurn(502, "gemini_audio_too_long");
      return;
    }

    if (!activeTurn.res.headersSent) {
      activeTurn.res.writeHead(200, {
        "content-type": "audio/pcm;rate=24000",
        "cache-control": "no-store",
        connection: "close",
        "x-atlas-relay-version": VERSION,
        "x-atlas-model": sessionModel,
        "x-atlas-voice": VOICE,
      });
      console.log(
        `First Gemini audio after ${Date.now() - activeTurn.submittedAt} ms.`,
      );
    }

    activeTurn.outputBytes += pcm.length;
    activeTurn.res.write(pcm);
  }

  if (content.generationComplete) {
    console.log("Gemini generation complete; waiting for true turnComplete.");
  }

  if (content.turnComplete) {
    scheduleFinish("turnComplete");
  }
}

async function connectModel(model) {
  await loadSdk();
  if (!ai) ai = new GoogleGenAI({ apiKey: API_KEY });

  const serial = ++sessionSerial;
  const config = {
    responseModalities: [Modality.AUDIO],
    systemInstruction: SYSTEM_INSTRUCTION,
    realtimeInputConfig: {
      automaticActivityDetection: { disabled: true },
    },
    speechConfig: {
      voiceConfig: {
        prebuiltVoiceConfig: { voiceName: VOICE },
      },
    },
  };
  if (model.includes("3.1")) config.thinkingConfig = { thinkingLevel: "minimal" };

  console.log(`Opening persistent Gemini Live session: ${model}`);
  const next = await ai.live.connect({
    model,
    config,
    callbacks: {
      onopen() {
        if (serial === sessionSerial) console.log(`Gemini Live opened: ${model}`);
      },
      onmessage(message) {
        handleGeminiMessage(message, serial);
      },
      onerror(event) {
        if (serial !== sessionSerial) return;
        const detail = event?.message || String(event);
        console.error(`Gemini Live error: ${detail}`);
        if (activeTurn) failTurn(502, `gemini_error: ${detail}`);
        else {
          session = null;
          sessionModel = "";
          scheduleReconnect("socket error");
        }
      },
      onclose(event) {
        if (serial !== sessionSerial) return;
        const code = event?.code ?? 1000;
        const reason = event?.reason || "";
        console.log(`Gemini Live closed: ${code} ${reason}`);
        session = null;
        sessionModel = "";
        if (activeTurn) failTurn(503, `gemini_session_closed_${code}`, false);
        scheduleReconnect("socket closed");
      },
    },
  });

  if (serial !== sessionSerial) {
    try { next.close(); } catch {}
    throw new Error("superseded Gemini connection");
  }

  session = next;
  sessionModel = model;
  sessionConnections += process.env.ATLAS_MOCK_GEMINI === "1" ? 0 : 1;
  console.log(`Persistent Gemini Live ready: ${model}`);
  return next;
}

async function ensureSession(force = false) {
  if (!API_KEY) throw new Error("missing_GEMINI_API_KEY");
  if (session && !force) return session;
  if (connectPromise) return connectPromise;
  if (session && force) clearSession("forced reconnect");

  connectPromise = (async () => {
    let lastError;
    for (const model of [...new Set([MODEL, FALLBACK_MODEL].filter(Boolean))]) {
      try {
        return await connectModel(model);
      } catch (error) {
        lastError = error;
        console.error(`Gemini connection failed (${model}):`, error.message);
      }
    }
    throw lastError || new Error("no_live_model_available");
  })();

  try {
    return await connectPromise;
  } finally {
    connectPromise = null;
  }
}

function readBody(req) {
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
    req.on("aborted", () => fail(Object.assign(new Error("client_aborted_upload"), { status: 499 })));
    req.on("error", (error) => fail(Object.assign(error, { status: 400 })));
  });
}

function submit(audio, res) {
  if (!session) throw Object.assign(new Error("gemini_not_ready"), { status: 503 });
  if (activeTurn) throw Object.assign(new Error("atlas_busy"), { status: 409 });

  const turn = {
    res,
    startedAt: Date.now(),
    submittedAt: 0,
    inputBytes: audio.length,
    outputBytes: 0,
    finished: false,
    timeout: null,
    finishTimer: null,
  };
  activeTurn = turn;

  try {
    session.sendRealtimeInput({ activityStart: {} });
    for (let offset = 0; offset < audio.length; offset += INPUT_CHUNK_BYTES) {
      const chunk = audio.subarray(offset, Math.min(offset + INPUT_CHUNK_BYTES, audio.length));
      session.sendRealtimeInput({
        audio: {
          data: chunk.toString("base64"),
          mimeType: "audio/pcm;rate=16000",
        },
      });
    }
    session.sendRealtimeInput({ activityEnd: {} });
    turn.submittedAt = Date.now();
    console.log(
      `PCM submitted to persistent Gemini session. bytes=${audio.length} ` +
        `model=${sessionModel} session=${sessionSerial}`,
    );
    turn.timeout = setTimeout(() => {
      if (activeTurn === turn) failTurn(504, "gemini_reply_timeout");
    }, TURN_TIMEOUT_MS);
  } catch (error) {
    failTurn(502, `gemini_submit_failed: ${error.message}`);
  }
}

async function handleTurn(req, res, parsed) {
  if (!DEVICE_TOKEN || !API_KEY) {
    json(res, 503, {
      ok: false,
      error: "relay_not_configured",
      hasGeminiKey: Boolean(API_KEY),
      hasDeviceToken: Boolean(DEVICE_TOKEN),
    });
    return;
  }
  if (!authorized(parsed)) {
    json(res, 401, { ok: false, error: "unauthorized" });
    return;
  }
  if (activeTurn || requestInProgress) {
    json(res, 409, { ok: false, error: "atlas_busy" });
    return;
  }

  requestInProgress = true;
  console.log("Atlas HTTP turn request received.");
  try {
    const ready = ensureSession(false);
    const audio = await readBody(req);
    await ready;
    if (!audio.length) throw Object.assign(new Error("empty_audio"), { status: 400 });
    if (audio.length % 2) throw Object.assign(new Error("odd_pcm_length"), { status: 400 });
    console.log(`Atlas HTTP upload received. Bytes: ${audio.length}`);
    submit(audio, res);
  } catch (error) {
    const status = error?.status || 503;
    const message = error?.message || String(error);
    console.error(`Atlas HTTP request failed: ${message}`);
    if (!res.headersSent) json(res, status, { ok: false, error: message });
    else if (!res.writableEnded) res.destroy();
  } finally {
    requestInProgress = false;
  }
}

const server = http.createServer(async (req, res) => {
  let parsed;
  try {
    parsed = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  } catch {
    json(res, 400, { ok: false, error: "bad_url" });
    return;
  }

  if (req.method === "GET" && (parsed.pathname === "/" || parsed.pathname === "/health")) {
    json(res, 200, {
      ok: true,
      service: "atlas-live-relay",
      version: VERSION,
      mode: "proven-http-turn-persistent-gemini-only",
      configured: Boolean(API_KEY && DEVICE_TOKEN),
      geminiReady: Boolean(session),
      geminiConnecting: Boolean(connectPromise),
      activeModel: sessionModel || null,
      voice: VOICE,
      busy: Boolean(activeTurn || requestInProgress),
      completedTurns,
      sessionConnections,
    });
    return;
  }

  if (req.method === "GET" && parsed.pathname === "/warm") {
    if (!authorized(parsed)) {
      json(res, 401, { ok: false, error: "unauthorized" });
      return;
    }
    try {
      await ensureSession(false);
      json(res, 200, { ok: true, geminiReady: true, model: sessionModel });
    } catch (error) {
      json(res, 503, { ok: false, error: error.message });
    }
    return;
  }

  if (req.method === "POST" && parsed.pathname === "/turn") {
    await handleTurn(req, res, parsed);
    return;
  }

  json(res, 404, { ok: false, error: "not_found" });
});

server.keepAliveTimeout = 65_000;
server.headersTimeout = 70_000;
server.requestTimeout = 0;

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Atlas Relay ${VERSION} listening on 0.0.0.0:${PORT}`);
  console.log("ESP32: proven HTTPS POST /turn per completed recording");
  console.log("Gemini: one persistent Live session reused between turns");
  console.log(`Configured: ${Boolean(API_KEY && DEVICE_TOKEN)}`);

  if (API_KEY) {
    void ensureSession(false).catch((error) => {
      console.error("Initial Gemini warm-up failed:", error.message);
      scheduleReconnect("initial warm-up");
    });
  }
});
