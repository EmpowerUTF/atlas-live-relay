import http from "node:http";
import { URL } from "node:url";
import { GoogleGenAI, Modality } from "@google/genai";

const PORT = Number(process.env.PORT || 8080);
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const ATLAS_DEVICE_TOKEN = process.env.ATLAS_DEVICE_TOKEN || "";
const GEMINI_MODEL =
  process.env.GEMINI_MODEL || "gemini-3.1-flash-live-preview";
const ATLAS_VOICE = process.env.ATLAS_VOICE || "Kore";
const SYSTEM_INSTRUCTION =
  process.env.ATLAS_SYSTEM_INSTRUCTION ||
  "You are Atlas, a fast, practical AI companion. Speak naturally and concisely. " +
    "Answer directly. Keep routine answers short unless detail is requested.";

const RELAY_VERSION = "2.2.0-buffered-response";
const INPUT_FRAME_BYTES = 3200; // 100 ms of 16 kHz mono PCM16
const MAX_INPUT_BYTES = 192000; // 6 seconds maximum
const TURN_TIMEOUT_MS = 30000;
const MAX_OUTPUT_BYTES = 384000; // 8 seconds of 24 kHz mono PCM16

if (!GEMINI_API_KEY) {
  console.error("Missing required environment variable: GEMINI_API_KEY");
  process.exit(1);
}

if (!ATLAS_DEVICE_TOKEN) {
  console.error("Missing required environment variable: ATLAS_DEVICE_TOKEN");
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

function safeJson(res, status, payload) {
  if (res.writableEnded) return;
  res.writeHead(status, {
    "content-type": "application/json",
    "cache-control": "no-store",
    connection: "close",
  });
  res.end(JSON.stringify(payload));
}

function collectRequestBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let total = 0;
    let finished = false;

    const fail = (error) => {
      if (finished) return;
      finished = true;
      reject(error);
    };

    req.on("data", (chunk) => {
      if (finished) return;
      total += chunk.length;

      if (total > MAX_INPUT_BYTES) {
        fail(Object.assign(new Error("input_too_large"), { status: 413 }));
        req.destroy();
        return;
      }

      chunks.push(Buffer.from(chunk));
    });

    req.on("end", () => {
      if (finished) return;
      finished = true;
      resolve(Buffer.concat(chunks, total));
    });

    req.on("aborted", () => {
      fail(Object.assign(new Error("client_aborted_upload"), { status: 499 }));
    });

    req.on("error", (error) => {
      fail(Object.assign(new Error(`request_error: ${error.message}`), { status: 400 }));
    });
  });
}

async function runGeminiTurn(audio, res) {
  let session = null;
  let settled = false;
  let timeout = null;
  let outputBytes = 0;
  const outputChunks = [];

  const closeSession = () => {
    try {
      session?.close();
    } catch (error) {
      console.error("Gemini session close error:", error?.message || error);
    }
  };

  const finish = () => {
    if (settled) return;
    settled = true;

    if (timeout) {
      clearTimeout(timeout);
      timeout = null;
    }

    if (outputBytes === 0) {
      safeJson(res, 502, {
        ok: false,
        error: "gemini_returned_no_audio",
      });
      closeSession();
      return;
    }

    const completeAudio = Buffer.concat(outputChunks, outputBytes);

    res.writeHead(200, {
      "content-type": "audio/pcm;rate=24000",
      "content-length": String(completeAudio.length),
      "cache-control": "no-store",
      connection: "close",
      "x-atlas-relay-version": RELAY_VERSION,
      "x-atlas-voice": ATLAS_VOICE,
      "x-atlas-audio-bytes": String(completeAudio.length),
    });

    res.end(completeAudio);

    console.log(
      `Atlas HTTP buffered response sent. Input bytes: ${audio.length}, output bytes: ${outputBytes}`,
    );

    setTimeout(closeSession, 50);
  };

  const fail = (status, message) => {
    if (settled) return;
    settled = true;

    if (timeout) {
      clearTimeout(timeout);
      timeout = null;
    }

    console.error(`Atlas HTTP turn failed: ${message}`);

    if (!res.writableEnded) {
      if (!res.headersSent) {
        safeJson(res, status, { ok: false, error: message });
      } else {
        res.destroy();
      }
    }

    closeSession();
  };

  res.on("close", () => {
    if (!settled && !res.writableEnded) {
      settled = true;
      if (timeout) clearTimeout(timeout);
      console.log("Atlas HTTP client disconnected before buffered response was sent.");
      closeSession();
    }
  });

  try {
    session = await ai.live.connect({
      model: GEMINI_MODEL,
      config: {
        responseModalities: [Modality.AUDIO],
        systemInstruction: SYSTEM_INSTRUCTION,
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: {
              voiceName: ATLAS_VOICE,
            },
          },
        },
        realtimeInputConfig: {
          automaticActivityDetection: {
            disabled: true,
          },
        },
      },
      callbacks: {
        onopen: () => {
          console.log("HTTP turn Gemini WebSocket opened.");
        },

        onmessage: (message) => {
          if (settled) return;

          const content = message?.serverContent;

          if (content?.interrupted) {
            fail(409, "gemini_interrupted");
            return;
          }

          const parts = content?.modelTurn?.parts || [];

          for (const part of parts) {
            const inline = part.inlineData;
            if (!inline?.data) continue;

            const mimeType = inline.mimeType || "";
            if (!mimeType.startsWith("audio/pcm")) continue;

            const pcm = Buffer.from(inline.data, "base64");
            if (pcm.length === 0) continue;

            if (outputBytes + pcm.length > MAX_OUTPUT_BYTES) {
              fail(502, "gemini_audio_too_long");
              return;
            }

            outputChunks.push(pcm);
            outputBytes += pcm.length;
          }

          if (content?.generationComplete) {
            console.log(
              `HTTP turn Gemini generation complete. Buffered output bytes: ${outputBytes}`,
            );
          }

          if (content?.turnComplete) {
            finish();
          }

          if (message?.goAway) {
            console.log("HTTP turn Gemini sent goAway.");
          }
        },

        onerror: (event) => {
          const detail = event?.message || String(event);
          fail(502, `gemini_error: ${detail}`);
        },

        onclose: (event) => {
          const code = event?.code ?? 1000;
          const reason = event?.reason || "";
          console.log(`HTTP turn Gemini closed: ${code} ${reason}`);

          if (!settled) {
            fail(503, `gemini_session_closed_${code}`);
          }
        },
      },
    });

    console.log("HTTP turn Gemini session ready.");

    timeout = setTimeout(() => {
      fail(504, "gemini_turn_timeout");
    }, TURN_TIMEOUT_MS);

    session.sendRealtimeInput({ activityStart: {} });

    for (let offset = 0; offset < audio.length; offset += INPUT_FRAME_BYTES) {
      const frame = audio.subarray(
        offset,
        Math.min(offset + INPUT_FRAME_BYTES, audio.length),
      );

      session.sendRealtimeInput({
        audio: {
          data: frame.toString("base64"),
          mimeType: "audio/pcm;rate=16000",
        },
      });

      if ((offset / INPUT_FRAME_BYTES) % 5 === 4) {
        await new Promise((resolve) => setImmediate(resolve));
      }
    }

    session.sendRealtimeInput({ activityEnd: {} });
    console.log(`Atlas HTTP audio submitted. Input bytes: ${audio.length}`);
  } catch (error) {
    fail(502, `gemini_connect_failed: ${error?.message || error}`);
  }
}

const server = http.createServer(async (req, res) => {
  let parsed;

  try {
    parsed = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  } catch {
    safeJson(res, 400, { ok: false, error: "bad_url" });
    return;
  }

  if (
    req.method === "GET" &&
    (parsed.pathname === "/" || parsed.pathname === "/health")
  ) {
    safeJson(res, 200, {
      ok: true,
      service: "atlas-live-relay",
      version: RELAY_VERSION,
      model: GEMINI_MODEL,
      voice: ATLAS_VOICE,
      geminiReady: true,
      geminiMode: "on-demand-per-turn",
      responseMode: "buffered-content-length",
    });
    return;
  }

  if (req.method === "POST" && parsed.pathname === "/turn") {
    if (parsed.searchParams.get("token") !== ATLAS_DEVICE_TOKEN) {
      safeJson(res, 401, { ok: false, error: "unauthorized" });
      return;
    }

    console.log("Atlas HTTP turn request received.");

    try {
      const audio = await collectRequestBody(req);

      if (audio.length === 0) {
        safeJson(res, 400, { ok: false, error: "empty_audio" });
        return;
      }

      console.log(`Atlas HTTP upload received. Bytes: ${audio.length}`);
      await runGeminiTurn(audio, res);
    } catch (error) {
      const status = error?.status || 400;
      const message = error?.message || String(error);
      console.error(`Atlas HTTP request failed: ${message}`);
      safeJson(res, status, { ok: false, error: message });
    }

    return;
  }

  safeJson(res, 404, { ok: false, error: "not_found" });
});

server.keepAliveTimeout = 5000;
server.headersTimeout = 15000;
server.requestTimeout = 30000;

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Atlas HTTP Turn Relay listening on port ${PORT}`);
  console.log(`Relay version: ${RELAY_VERSION}`);
  console.log(`Model: ${GEMINI_MODEL}`);
  console.log(`Voice: ${ATLAS_VOICE}`);
  console.log("Gemini sessions open only when Atlas submits a voice turn.");
});
