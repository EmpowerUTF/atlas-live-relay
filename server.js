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

const RELAY_VERSION = "2.3.0-stream-diagnostics";
const INPUT_FRAME_BYTES = 3200; // 100 ms of 16 kHz mono PCM16
const MAX_INPUT_BYTES = 192000; // 6 seconds maximum
const TURN_TIMEOUT_MS = 25000;
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
  const blockAlign = 2;

  header.write("RIFF", 0, "ascii");
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVE", 8, "ascii");
  header.write("fmt ", 12, "ascii");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
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
      fail(Object.assign(new Error(`request_error: ${error.message}`), { status: 400 }));
    });
  });
}

async function openGeminiTurn(res) {
  let session = null;
  let settled = false;
  let timeout = null;
  let headersSent = false;
  let outputBytes = 0;
  const outputChunks = [];
  let inputTranscript = "";
  let outputTranscript = "";

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
    rejectDone(new Error(message));
  };

  const finish = () => {
    if (settled) return;
    settled = true;

    if (timeout) {
      clearTimeout(timeout);
      timeout = null;
    }

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

    if (!res.writableEnded) {
      res.end();
    }

    const outputPcm = Buffer.concat(outputChunks, outputBytes);

    lastTurn = {
      createdAt: new Date().toISOString(),
      inputPcm: lastTurn.inputPcm,
      outputPcm,
      inputTranscript: inputTranscript.trim(),
      outputTranscript: outputTranscript.trim(),
    };

    console.log(
      `Atlas HTTP streamed response complete. Output bytes: ${outputBytes}`,
    );
    console.log(
      `INPUT TRANSCRIPT: ${lastTurn.inputTranscript || "<empty>"}`,
    );
    console.log(
      `OUTPUT TRANSCRIPT: ${lastTurn.outputTranscript || "<empty>"}`,
    );

    closeSession();
    resolveDone({
      outputBytes,
      inputTranscript: lastTurn.inputTranscript,
      outputTranscript: lastTurn.outputTranscript,
    });
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
      thinkingConfig: {
        thinkingLevel: "minimal",
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

          if (!res.write(pcm)) {
            // Node will resume flushing automatically. The complete output is
            // still retained for the diagnostic WAV endpoint.
          }
        }

        if (content?.generationComplete) {
          console.log(
            `HTTP turn Gemini generation complete. Output bytes so far: ${outputBytes}`,
          );
        }

        if (content?.turnComplete) {
          finish();
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

  return {
    async submit(audio) {
      lastTurn = {
        createdAt: new Date().toISOString(),
        inputPcm: Buffer.from(audio),
        outputPcm: Buffer.alloc(0),
        inputTranscript: "",
        outputTranscript: "",
      };

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

      session.sendRealtimeInput({ audioStreamEnd: true });
      console.log(`Atlas HTTP audio submitted. Input bytes: ${audio.length}`);
    },
    done,
    close: closeSession,
  };
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
      geminiMode: "connect-during-upload",
      responseMode: "streamed-chunked",
      inputTranscription: true,
      outputTranscription: true,
      diagnostics: true,
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

    let turn = null;

    try {
      // Open Gemini while Atlas is still uploading the microphone turn. This
      // hides most of the Gemini session startup time behind recording.
      const turnPromise = openGeminiTurn(res);
      const audioPromise = collectRequestBody(req);
      turn = await turnPromise;
      const audio = await audioPromise;

      if (audio.length === 0) {
        turn.close();
        safeJson(res, 400, { ok: false, error: "empty_audio" });
        return;
      }

      console.log(`Atlas HTTP upload received. Bytes: ${audio.length}`);
      await turn.submit(audio);
      await turn.done;
    } catch (error) {
      const status = error?.status || 500;
      const message = error?.message || String(error);
      console.error(`Atlas HTTP request failed: ${message}`);

      try {
        turn?.close();
      } catch {
        // Ignore close failures during error cleanup.
      }

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
  console.log("Gemini connects while Atlas uploads each voice turn.");
});
