import http from "node:http";
import { URL } from "node:url";
import { GoogleGenAI, Modality } from "@google/genai";
import { WebSocket, WebSocketServer } from "ws";

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

const RELAY_VERSION = "2.0.0-http-turn";
const INPUT_FRAME_BYTES = 3200; // 100 ms, 16 kHz, mono PCM16
const MAX_INPUT_BYTES = 192000; // 6 seconds at 16 kHz PCM16
const TURN_TIMEOUT_MS = 30000;

if (!GEMINI_API_KEY) {
  console.error("Missing required environment variable: GEMINI_API_KEY");
  process.exit(1);
}

if (!ATLAS_DEVICE_TOKEN) {
  console.error("Missing required environment variable: ATLAS_DEVICE_TOKEN");
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

let session = null;
let geminiReady = false;
let connectInFlight = false;
let reconnectTimer = null;
let activeTurn = null;

function safeJson(res, status, payload) {
  if (res.writableEnded) return;
  res.writeHead(status, {
    "content-type": "application/json",
    "cache-control": "no-store",
    connection: "close",
  });
  res.end(JSON.stringify(payload));
}

function clearTurnTimeout(turn) {
  if (turn?.timeout) {
    clearTimeout(turn.timeout);
    turn.timeout = null;
  }
}

function finishActiveTurn() {
  if (!activeTurn) return;

  const turn = activeTurn;
  activeTurn = null;
  clearTurnTimeout(turn);

  if (!turn.res.writableEnded) {
    if (!turn.headersSent) {
      safeJson(turn.res, 502, {
        ok: false,
        error: "gemini_returned_no_audio",
      });
    } else {
      turn.res.end();
    }
  }

  console.log(
    `Turn complete. Input bytes: ${turn.inputBytes}, output bytes: ${turn.outputBytes}`,
  );
}

function failActiveTurn(status, message) {
  if (!activeTurn) return;

  const turn = activeTurn;
  activeTurn = null;
  clearTurnTimeout(turn);

  console.error(`Turn failed: ${message}`);

  if (!turn.res.writableEnded) {
    if (!turn.headersSent) {
      safeJson(turn.res, status, { ok: false, error: message });
    } else {
      turn.res.destroy();
    }
  }
}

function scheduleGeminiReconnect() {
  if (reconnectTimer || connectInFlight) return;

  geminiReady = false;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connectGemini();
  }, 1000);
}

function handleGeminiMessage(message) {
  const content = message?.serverContent;

  if (content?.interrupted && activeTurn) {
    failActiveTurn(409, "gemini_interrupted");
    return;
  }

  const parts = content?.modelTurn?.parts || [];

  for (const part of parts) {
    const inline = part.inlineData;
    if (!inline?.data || !activeTurn) continue;

    const mimeType = inline.mimeType || "";
    if (!mimeType.startsWith("audio/pcm")) continue;

    const pcm = Buffer.from(inline.data, "base64");
    if (pcm.length === 0) continue;

    if (!activeTurn.headersSent) {
      activeTurn.res.writeHead(200, {
        "content-type": "audio/pcm;rate=24000",
        "cache-control": "no-store",
        connection: "close",
        "x-atlas-relay-version": RELAY_VERSION,
        "x-atlas-voice": ATLAS_VOICE,
      });
      activeTurn.headersSent = true;
    }

    activeTurn.outputBytes += pcm.length;
    activeTurn.res.write(pcm);
  }

  if (content?.generationComplete && activeTurn) {
    console.log(
      `Gemini generation complete. Output bytes so far: ${activeTurn.outputBytes}`,
    );
  }

  if (content?.turnComplete && activeTurn) {
    finishActiveTurn();
  }

  if (message?.goAway) {
    console.log("Gemini sent goAway; reconnecting session.");
    scheduleGeminiReconnect();
  }
}

async function connectGemini() {
  if (connectInFlight) return;

  connectInFlight = true;

  try {
    const newSession = await ai.live.connect({
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
          console.log("Gemini SDK WebSocket opened.");
        },
        onmessage: (message) => {
          handleGeminiMessage(message);
        },
        onerror: (event) => {
          const detail = event?.message || String(event);
          console.error("Gemini SDK error:", detail);
          failActiveTurn(502, `gemini_error: ${detail}`);
        },
        onclose: (event) => {
          const code = event?.code ?? 1000;
          const reason = event?.reason || "";
          console.log(`Gemini SDK closed: ${code} ${reason}`);
          geminiReady = false;
          session = null;
          failActiveTurn(503, "gemini_session_closed");
          scheduleGeminiReconnect();
        },
      },
    });

    session = newSession;
    geminiReady = true;
    console.log("Gemini Live session ready through official SDK.");
  } catch (error) {
    const detail = error?.message || String(error);
    console.error("Gemini SDK connection failed:", detail);
    session = null;
    geminiReady = false;
    scheduleGeminiReconnect();
  } finally {
    connectInFlight = false;
  }
}

function forwardInputFrame(turn, frame) {
  if (!session || !geminiReady || frame.length === 0) return false;

  session.sendRealtimeInput({
    audio: {
      data: frame.toString("base64"),
      mimeType: "audio/pcm;rate=16000",
    },
  });

  turn.inputBytes += frame.length;
  return true;
}

function handleTurnRequest(req, res) {
  if (!geminiReady || !session) {
    safeJson(res, 503, { ok: false, error: "gemini_not_ready" });
    return;
  }

  if (activeTurn) {
    safeJson(res, 409, { ok: false, error: "turn_already_active" });
    return;
  }

  const turn = {
    req,
    res,
    headersSent: false,
    inputBytes: 0,
    outputBytes: 0,
    inputBuffer: Buffer.alloc(0),
    requestEnded: false,
    timeout: null,
  };

  activeTurn = turn;

  try {
    session.sendRealtimeInput({ activityStart: {} });
    console.log("Atlas HTTP turn started.");
  } catch (error) {
    failActiveTurn(503, `activity_start_failed: ${error?.message || error}`);
    return;
  }

  req.on("data", (chunk) => {
    if (activeTurn !== turn) return;

    if (turn.inputBytes + turn.inputBuffer.length + chunk.length > MAX_INPUT_BYTES) {
      failActiveTurn(413, "input_too_large");
      req.destroy();
      return;
    }

    turn.inputBuffer = Buffer.concat([turn.inputBuffer, chunk]);

    while (turn.inputBuffer.length >= INPUT_FRAME_BYTES) {
      const frame = turn.inputBuffer.subarray(0, INPUT_FRAME_BYTES);
      turn.inputBuffer = turn.inputBuffer.subarray(INPUT_FRAME_BYTES);

      try {
        if (!forwardInputFrame(turn, frame)) {
          failActiveTurn(503, "gemini_not_ready_during_upload");
          return;
        }
      } catch (error) {
        failActiveTurn(502, `audio_forward_failed: ${error?.message || error}`);
        return;
      }
    }
  });

  req.on("end", () => {
    if (activeTurn !== turn) return;
    turn.requestEnded = true;

    try {
      if (turn.inputBuffer.length > 0) {
        if (!forwardInputFrame(turn, turn.inputBuffer)) {
          failActiveTurn(503, "gemini_not_ready_during_final_upload");
          return;
        }
        turn.inputBuffer = Buffer.alloc(0);
      }

      session.sendRealtimeInput({ activityEnd: {} });
      console.log(`Atlas HTTP upload complete. Input bytes: ${turn.inputBytes}`);

      turn.timeout = setTimeout(() => {
        if (activeTurn === turn) {
          failActiveTurn(504, "gemini_turn_timeout");
        }
      }, TURN_TIMEOUT_MS);
    } catch (error) {
      failActiveTurn(502, `activity_end_failed: ${error?.message || error}`);
    }
  });

  req.on("aborted", () => {
    if (activeTurn === turn) {
      failActiveTurn(499, "client_aborted_upload");
    }
  });

  req.on("error", (error) => {
    if (activeTurn === turn) {
      failActiveTurn(400, `request_error: ${error.message}`);
    }
  });

  res.on("close", () => {
    if (activeTurn === turn && !res.writableEnded) {
      failActiveTurn(499, "client_disconnected_during_response");
    }
  });
}

const server = http.createServer((req, res) => {
  let parsed;

  try {
    parsed = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  } catch {
    safeJson(res, 400, { ok: false, error: "bad_url" });
    return;
  }

  if (req.method === "GET" && (parsed.pathname === "/" || parsed.pathname === "/health")) {
    safeJson(res, 200, {
      ok: true,
      service: "atlas-live-relay",
      version: RELAY_VERSION,
      model: GEMINI_MODEL,
      voice: ATLAS_VOICE,
      geminiReady,
    });
    return;
  }

  if (req.method === "POST" && parsed.pathname === "/turn") {
    if (parsed.searchParams.get("token") !== ATLAS_DEVICE_TOKEN) {
      safeJson(res, 401, { ok: false, error: "unauthorized" });
      return;
    }

    handleTurnRequest(req, res);
    return;
  }

  safeJson(res, 404, { ok: false, error: "not_found" });
});

// Legacy /atlas WebSocket endpoint remains available during migration.
const atlasWss = new WebSocketServer({
  noServer: true,
  maxPayload: 128 * 1024,
  perMessageDeflate: false,
});

server.on("upgrade", (req, socket, head) => {
  let parsed;

  try {
    parsed = new URL(req.url, `http://${req.headers.host}`);
  } catch {
    socket.destroy();
    return;
  }

  if (parsed.pathname !== "/atlas") {
    socket.destroy();
    return;
  }

  if (parsed.searchParams.get("token") !== ATLAS_DEVICE_TOKEN) {
    socket.write("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
    socket.destroy();
    return;
  }

  atlasWss.handleUpgrade(req, socket, head, (ws) => {
    atlasWss.emit("connection", ws, req);
  });
});

atlasWss.on("connection", (atlas) => {
  console.log("Atlas device connected.");

  try {
    atlas._socket?.setNoDelay(true);
    atlas._socket?.setKeepAlive(true, 10000);
  } catch {
    // Socket tuning is best effort only.
  }

  const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

  let session = null;
  let geminiReady = false;
  let closing = false;
  let connectInFlight = false;
  let reconnectTimer = null;
  let inputAudioBytes = 0;
  let outputAudioBytes = 0;
  let outputAudioSentBytes = 0;
  let outputQueue = [];
  let outputQueueOffset = 0;
  let pendingTurnComplete = false;

  const OUTPUT_FRAME_BYTES = 960; // 20 ms of 24 kHz mono PCM16
  const OUTPUT_FRAME_MS = 20;
  const MAX_ATLAS_BUFFERED_BYTES = 64 * 1024;

  const sendAtlasJson = (payload) => {
    if (atlas.readyState === WebSocket.OPEN) {
      atlas.send(JSON.stringify(payload));
    }
  };

  const clearOutputQueue = () => {
    outputQueue = [];
    outputQueueOffset = 0;
    outputAudioSentBytes = 0;
    pendingTurnComplete = false;
  };

  const queuedOutputBytes = () => {
    let total = 0;

    for (let i = 0; i < outputQueue.length; i += 1) {
      total += outputQueue[i].length;
    }

    return Math.max(0, total - outputQueueOffset);
  };

  const takeOutputFrame = (maximumBytes) => {
    if (outputQueue.length === 0) return null;

    const parts = [];
    let total = 0;
    let remaining = maximumBytes;

    while (remaining > 0 && outputQueue.length > 0) {
      const first = outputQueue[0];
      const available = first.length - outputQueueOffset;
      const take = Math.min(available, remaining);

      parts.push(first.subarray(outputQueueOffset, outputQueueOffset + take));
      total += take;
      remaining -= take;
      outputQueueOffset += take;

      if (outputQueueOffset >= first.length) {
        outputQueue.shift();
        outputQueueOffset = 0;
      }
    }

    if (parts.length === 1) return parts[0];
    return Buffer.concat(parts, total);
  };

  const audioPacer = setInterval(() => {
    if (closing || atlas.readyState !== WebSocket.OPEN) return;

    if (atlas.bufferedAmount > MAX_ATLAS_BUFFERED_BYTES) {
      return;
    }

    const frame = takeOutputFrame(OUTPUT_FRAME_BYTES);

    if (frame && frame.length > 0) {
      atlas.send(frame, { binary: true });
      outputAudioSentBytes += frame.length;
      return;
    }

    if (pendingTurnComplete) {
      console.log(
        `Gemini audio delivered. Generated: ${outputAudioBytes}, sent: ${outputAudioSentBytes}`,
      );

      sendAtlasJson({ type: "turn_complete" });
      pendingTurnComplete = false;
      outputAudioBytes = 0;
      outputAudioSentBytes = 0;
    }
  }, OUTPUT_FRAME_MS);

  const pingTimer = setInterval(() => {
    if (!closing && atlas.readyState === WebSocket.OPEN) {
      try {
        atlas.ping();
      } catch {
        // The close handler will clean up if the socket has died.
      }
    }
  }, 20000);

  const scheduleGeminiReconnect = () => {
    if (closing || atlas.readyState !== WebSocket.OPEN || reconnectTimer) return;

    geminiReady = false;
    sendAtlasJson({ type: "gemini_reconnecting" });

    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connectGemini();
    }, 750);
  };

  const handleGeminiMessage = (message) => {
    const content = message?.serverContent;

    if (content) {
      if (content.interrupted) {
        console.log("Gemini response interrupted.");
        clearOutputQueue();
        sendAtlasJson({ type: "interrupted" });
      }

      const parts = content.modelTurn?.parts || [];

      for (const part of parts) {
        const inline = part.inlineData;
        if (!inline?.data) continue;

        const mimeType = inline.mimeType || "";
        if (!mimeType.startsWith("audio/pcm")) continue;

        const pcm = Buffer.from(inline.data, "base64");
        outputAudioBytes += pcm.length;
        outputQueue.push(pcm);
      }

      if (content.generationComplete) {
        console.log(
          `Gemini generation complete. Output bytes: ${outputAudioBytes}, queued: ${queuedOutputBytes()}`,
        );
        sendAtlasJson({ type: "generation_complete" });
      }

      if (content.turnComplete) {
        console.log(
          `Gemini turn complete. Output bytes: ${outputAudioBytes}, queued: ${queuedOutputBytes()}`,
        );
        pendingTurnComplete = true;
      }
    }

    if (message?.goAway) {
      console.log("Gemini sent goAway.");
      sendAtlasJson({ type: "go_away", detail: message.goAway });
    }
  };

  const connectGemini = async () => {
    if (closing || connectInFlight || atlas.readyState !== WebSocket.OPEN) return;

    connectInFlight = true;

    try {
      const newSession = await ai.live.connect({
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
            console.log("Gemini SDK WebSocket opened.");
          },

          onmessage: (message) => {
            handleGeminiMessage(message);
          },

          onerror: (event) => {
            const detail = event?.message || String(event);
            console.error("Gemini SDK error:", detail);
            sendAtlasJson({
              type: "error",
              source: "gemini",
              message: detail,
            });
          },

          onclose: (event) => {
            const code = event?.code ?? 1000;
            const reason = event?.reason || "";
            console.log(`Gemini SDK closed: ${code} ${reason}`);
            geminiReady = false;
            session = null;

            if (!closing) {
              scheduleGeminiReconnect();
            }
          },
        },
      });

      if (closing) {
        newSession.close();
        return;
      }

      session = newSession;
      geminiReady = true;
      inputAudioBytes = 0;
      outputAudioBytes = 0;
      clearOutputQueue();
      console.log("Gemini Live session ready through official SDK.");

      sendAtlasJson({
        type: "ready",
        model: GEMINI_MODEL,
        voice: ATLAS_VOICE,
      });
    } catch (error) {
      const detail = error?.message || String(error);
      console.error("Gemini SDK connection failed:", detail);
      sendAtlasJson({
        type: "error",
        source: "gemini",
        message: detail,
      });
      scheduleGeminiReconnect();
    } finally {
      connectInFlight = false;
    }
  };

  atlas.on("message", (data, isBinary) => {
    if (!geminiReady || !session) return;

    try {
      if (isBinary) {
        const pcm = Buffer.from(data);
        inputAudioBytes += pcm.length;

        session.sendRealtimeInput({
          audio: {
            data: pcm.toString("base64"),
            mimeType: "audio/pcm;rate=16000",
          },
        });
        return;
      }

      let control;
      try {
        control = JSON.parse(data.toString());
      } catch {
        return;
      }

      if (control.type === "activity_start") {
        inputAudioBytes = 0;
        outputAudioBytes = 0;
        clearOutputQueue();
        console.log("Atlas activity start.");
        session.sendRealtimeInput({ activityStart: {} });
      } else if (control.type === "activity_end") {
        console.log(`Atlas activity end. Input bytes: ${inputAudioBytes}`);
        session.sendRealtimeInput({ activityEnd: {} });
      } else if (control.type === "text" && typeof control.text === "string") {
        session.sendRealtimeInput({ text: control.text });
      } else if (control.type === "ping") {
        sendAtlasJson({ type: "pong", at: Date.now() });
      }
    } catch (error) {
      const detail = error?.message || String(error);
      console.error("Relay input forwarding error:", detail);
      sendAtlasJson({
        type: "error",
        source: "relay",
        message: detail,
      });
    }
  });

  atlas.on("error", (error) => {
    console.error("Atlas WebSocket error:", error.message);
  });

  atlas.on("close", (code, reasonBuffer) => {
    const reason = reasonBuffer?.toString?.() || "";
    console.log(`Atlas device disconnected. Code: ${code} ${reason}`);
    closing = true;
    geminiReady = false;
    clearInterval(audioPacer);
    clearInterval(pingTimer);
    clearOutputQueue();

    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }

    try {
      session?.close();
    } catch (error) {
      console.error("Gemini session close error:", error?.message || error);
    }
  });

  connectGemini();
});


server.keepAliveTimeout = 5000;
server.headersTimeout = 10000;
server.requestTimeout = 15000;

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Atlas HTTP Turn Relay listening on port ${PORT}`);
  console.log(`Relay version: ${RELAY_VERSION}`);
  console.log(`Model: ${GEMINI_MODEL}`);
  console.log(`Voice: ${ATLAS_VOICE}`);
  connectGemini();
});
