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
  "You are Atlas, a fast practical AI companion. Respond only to the user's spoken request. " +
    "Do not initiate a greeting and do not say hello unless the user greeted you. " +
    "If the speech is unclear, say exactly: I didn't catch that. " +
    "Keep routine answers to one or two concise sentences.";

const RELAY_VERSION = "2.4.0-manual-turn";
const MAX_INPUT_BYTES = 192000; // 6 s of 16 kHz mono PCM16
const MAX_OUTPUT_BYTES = 384000; // 8 s of 24 kHz mono PCM16
const GEMINI_REPLY_TIMEOUT_MS = 20000;

if (!GEMINI_API_KEY) {
  console.error("Missing required environment variable: GEMINI_API_KEY");
  process.exit(1);
}

if (!ATLAS_DEVICE_TOKEN) {
  console.error("Missing required environment variable: ATLAS_DEVICE_TOKEN");
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

let lastTurn = {
  createdAt: null,
  inputPcm: Buffer.alloc(0),
  outputPcm: Buffer.alloc(0),
  inputTranscript: "",
  outputTranscript: "",
};

function safeJson(res, status, payload) {
  if (res.writableEnded) return;
  res.writeHead(status, {
    "content-type": "application/json",
    "cache-control": "no-store",
    connection: "close",
  });
  res.end(JSON.stringify(payload));
}

function authorized(parsed) {
  return parsed.searchParams.get("token") === ATLAS_DEVICE_TOKEN;
}

function wavFromPcm(pcm, sampleRate) {
  const header = Buffer.alloc(44);
  const byteRate = sampleRate * 2;

  header.write("RIFF", 0, "ascii");
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVE", 8, "ascii");
  header.write("fmt ", 12, "ascii");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36, "ascii");
  header.writeUInt32LE(pcm.length, 40);

  return Buffer.concat([header, pcm]);
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
      fail(
        Object.assign(new Error(`request_error: ${error.message}`), {
          status: 400,
        }),
      );
    });
  });
}

async function runGeminiTurn(audio, res) {
  let session = null;
  let settled = false;
  let timeout = null;
  let headersSent = false;
  let outputBytes = 0;
  let firstAudioAt = 0;
  let inputTranscript = "";
  let outputTranscript = "";
  const outputChunks = [];
  const turnStartedAt = Date.now();

  let resolveDone;
  let rejectDone;
  const done = new Promise((resolve, reject) => {
    resolveDone = resolve;
    rejectDone = reject;
  });

  const closeSession = () => {
    try {
      session?.close();
    } catch (error) {
      console.error("Gemini session close error:", error?.message || error);
    }
  };

  const fail = (status, message) => {
    if (settled) return;
    settled = true;

    if (timeout) clearTimeout(timeout);
    timeout = null;

    console.error(`Atlas HTTP turn failed: ${message}`);

    if (!res.writableEnded) {
      if (!res.headersSent) {
        safeJson(res, status, { ok: false, error: message });
      } else {
        res.destroy();
      }
    }

    closeSession();
    rejectDone(new Error(message));
  };

  const finish = (reason) => {
    if (settled) return;
    settled = true;

    if (timeout) clearTimeout(timeout);
    timeout = null;

    if (outputBytes === 0) {
      if (!res.headersSent) {
        safeJson(res, 502, { ok: false, error: "gemini_returned_no_audio" });
      } else if (!res.writableEnded) {
        res.end();
      }
      closeSession();
      rejectDone(new Error("gemini_returned_no_audio"));
      return;
    }

    if (!res.writableEnded) res.end();

    lastTurn = {
      createdAt: new Date().toISOString(),
      inputPcm: Buffer.from(audio),
      outputPcm: Buffer.concat(outputChunks, outputBytes),
      inputTranscript: inputTranscript.trim(),
      outputTranscript: outputTranscript.trim(),
    };

    console.log(
      `Atlas HTTP response complete (${reason}). Input bytes: ${audio.length}, output bytes: ${outputBytes}, total: ${Date.now() - turnStartedAt} ms`,
    );
    console.log(`INPUT TRANSCRIPT: ${lastTurn.inputTranscript || "<empty>"}`);
    console.log(`OUTPUT TRANSCRIPT: ${lastTurn.outputTranscript || "<empty>"}`);

    closeSession();
    resolveDone();
  };

  res.on("close", () => {
    if (!settled && !res.writableEnded) {
      fail(499, "client_disconnected_during_response");
    }
  });

  session = await ai.live.connect({
    model: GEMINI_MODEL,
    config: {
      responseModalities: [Modality.AUDIO],
      systemInstruction: SYSTEM_INSTRUCTION,
      inputAudioTranscription: {},
      outputAudioTranscription: {},
      realtimeInputConfig: {
        automaticActivityDetection: {
          disabled: true,
        },
      },
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: {
            voiceName: ATLAS_VOICE,
          },
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

        if (content?.inputTranscription?.text) {
          inputTranscript += content.inputTranscription.text;
        }

        if (content?.outputTranscription?.text) {
          outputTranscript += content.outputTranscription.text;
        }

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

          if (!headersSent) {
            headersSent = true;
            firstAudioAt = Date.now();
            console.log(
              `HTTP turn first Gemini audio after ${firstAudioAt - turnStartedAt} ms.`,
            );

            res.writeHead(200, {
              "content-type": "audio/pcm;rate=24000",
              "cache-control": "no-store",
              connection: "close",
              "x-atlas-relay-version": RELAY_VERSION,
              "x-atlas-voice": ATLAS_VOICE,
            });
          }

          outputChunks.push(pcm);
          outputBytes += pcm.length;
          res.write(pcm);
        }

        // generationComplete means every output audio chunk for this turn has
        // been generated. Ending here avoids waiting for Gemini's simulated
        // real-time playback delay before turnComplete.
        if (content?.generationComplete) {
          console.log(
            `HTTP turn Gemini generation complete. Output bytes: ${outputBytes}`,
          );
          finish("generationComplete");
          return;
        }

        if (content?.turnComplete) {
          finish("turnComplete");
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

  // This matches Google's documented custom-VAD sequence exactly:
  // disable automatic VAD, then activityStart -> audio -> activityEnd.
  session.sendRealtimeInput({ activityStart: {} });
  session.sendRealtimeInput({
    audio: {
      data: audio.toString("base64"),
      mimeType: "audio/pcm;rate=16000",
    },
  });
  session.sendRealtimeInput({ activityEnd: {} });

  console.log(`Atlas HTTP audio submitted. Input bytes: ${audio.length}`);

  timeout = setTimeout(() => {
    fail(504, "gemini_reply_timeout");
  }, GEMINI_REPLY_TIMEOUT_MS);

  await done;
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
      geminiMode: "on-demand-manual-vad",
      responseMode: "streamed-chunked",
      inputTranscription: true,
      outputTranscription: true,
    });
    return;
  }

  if (req.method === "GET" && parsed.pathname === "/debug/last-turn") {
    if (!authorized(parsed)) {
      safeJson(res, 401, { ok: false, error: "unauthorized" });
      return;
    }

    safeJson(res, 200, {
      ok: true,
      createdAt: lastTurn.createdAt,
      inputBytes: lastTurn.inputPcm.length,
      outputBytes: lastTurn.outputPcm.length,
      inputTranscript: lastTurn.inputTranscript,
      outputTranscript: lastTurn.outputTranscript,
    });
    return;
  }

  if (req.method === "GET" && parsed.pathname === "/debug/last-input.wav") {
    if (!authorized(parsed)) {
      safeJson(res, 401, { ok: false, error: "unauthorized" });
      return;
    }

    if (lastTurn.inputPcm.length === 0) {
      safeJson(res, 404, { ok: false, error: "no_input_audio_yet" });
      return;
    }

    const wav = wavFromPcm(lastTurn.inputPcm, 16000);
    res.writeHead(200, {
      "content-type": "audio/wav",
      "content-length": String(wav.length),
      "cache-control": "no-store",
      connection: "close",
    });
    res.end(wav);
    return;
  }

  if (req.method === "GET" && parsed.pathname === "/debug/last-output.wav") {
    if (!authorized(parsed)) {
      safeJson(res, 401, { ok: false, error: "unauthorized" });
      return;
    }

    if (lastTurn.outputPcm.length === 0) {
      safeJson(res, 404, { ok: false, error: "no_output_audio_yet" });
      return;
    }

    const wav = wavFromPcm(lastTurn.outputPcm, 24000);
    res.writeHead(200, {
      "content-type": "audio/wav",
      "content-length": String(wav.length),
      "cache-control": "no-store",
      connection: "close",
    });
    res.end(wav);
    return;
  }

  if (req.method === "POST" && parsed.pathname === "/turn") {
    if (!authorized(parsed)) {
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
      const status = error?.status || 500;
      const message = error?.message || String(error);
      console.error(`Atlas HTTP request failed: ${message}`);

      if (!res.headersSent) {
        safeJson(res, status, { ok: false, error: message });
      } else if (!res.writableEnded) {
        res.destroy();
      }
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
  console.log("Gemini sessions open only after Atlas finishes uploading a turn.");
});
