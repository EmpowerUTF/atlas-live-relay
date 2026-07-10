import http from "node:http";
import { URL } from "node:url";
import { WebSocket, WebSocketServer } from "ws";

const PORT = Number(process.env.PORT || 3000);
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const ATLAS_DEVICE_TOKEN = process.env.ATLAS_DEVICE_TOKEN || "";
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.1-flash-live-preview";
const ATLAS_VOICE = process.env.ATLAS_VOICE || "Kore";
const SYSTEM_INSTRUCTION =
  process.env.ATLAS_SYSTEM_INSTRUCTION ||
  "You are Atlas, a fast, practical AI companion. Speak naturally and concisely. " +
    "Answer the user's request directly. Keep routine answers short unless detail is requested.";

if (!GEMINI_API_KEY) {
  console.error("Missing required environment variable: GEMINI_API_KEY");
  process.exit(1);
}

if (!ATLAS_DEVICE_TOKEN) {
  console.error("Missing required environment variable: ATLAS_DEVICE_TOKEN");
  process.exit(1);
}

const server = http.createServer((req, res) => {
  if (req.url === "/" || req.url === "/health") {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        ok: true,
        service: "atlas-live-relay",
        version: "1.1.0",
        model: GEMINI_MODEL,
      }),
    );
    return;
  }

  res.writeHead(404, { "content-type": "application/json" });
  res.end(JSON.stringify({ ok: false, error: "not_found" }));
});

const atlasWss = new WebSocketServer({ noServer: true, maxPayload: 64 * 1024 });

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
  const geminiUrl =
    "wss://generativelanguage.googleapis.com/ws/" +
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent" +
    `?key=${encodeURIComponent(GEMINI_API_KEY)}`;

  const gemini = new WebSocket(geminiUrl, {
    handshakeTimeout: 15000,
    perMessageDeflate: false,
    maxPayload: 4 * 1024 * 1024,
  });

  let geminiReady = false;
  let closed = false;

  const sendAtlasJson = (payload) => {
    if (atlas.readyState === WebSocket.OPEN) {
      atlas.send(JSON.stringify(payload));
    }
  };

  const closeBoth = (code = 1000, reason = "session_closed") => {
    if (closed) return;
    closed = true;

    if (atlas.readyState === WebSocket.OPEN || atlas.readyState === WebSocket.CONNECTING) {
      atlas.close(code, reason);
    }

    if (gemini.readyState === WebSocket.OPEN || gemini.readyState === WebSocket.CONNECTING) {
      gemini.close(code, reason);
    }
  };

  gemini.on("open", () => {
    const setupMessage = {
      setup: {
        model: `models/${GEMINI_MODEL}`,
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
          parts: [{ text: SYSTEM_INSTRUCTION }],
        },
        realtimeInputConfig: {
          automaticActivityDetection: {
            disabled: false,
            startOfSpeechSensitivity: "START_SENSITIVITY_HIGH",
            endOfSpeechSensitivity: "END_SENSITIVITY_HIGH",
            prefixPaddingMs: 100,
            silenceDurationMs: 500,
          },
        },
        inputAudioTranscription: {},
        outputAudioTranscription: {},
      },
    };

    gemini.send(JSON.stringify(setupMessage));
  });

  gemini.on("message", (raw, isBinary) => {
    if (isBinary) {
      return;
    }

    let message;
    try {
      message = JSON.parse(raw.toString());
    } catch (error) {
      console.error("Could not parse Gemini message:", error.message);
      return;
    }

    if (message.setupComplete) {
      geminiReady = true;
      console.log("Gemini setup complete for Atlas.");
      sendAtlasJson({ type: "ready", model: GEMINI_MODEL, voice: ATLAS_VOICE });
      return;
    }

    const content = message.serverContent;
    if (content) {
      if (content.interrupted) {
        sendAtlasJson({ type: "interrupted" });
      }

      if (content.inputTranscription?.text) {
        sendAtlasJson({
          type: "input_transcript",
          text: content.inputTranscription.text,
        });
      }

      if (content.outputTranscription?.text) {
        sendAtlasJson({
          type: "output_transcript",
          text: content.outputTranscription.text,
        });
      }

      const parts = content.modelTurn?.parts || [];
      for (const part of parts) {
        const inline = part.inlineData;
        if (!inline?.data) continue;

        const mimeType = inline.mimeType || "";
        if (!mimeType.startsWith("audio/pcm")) continue;

        const pcm = Buffer.from(inline.data, "base64");
        if (atlas.readyState === WebSocket.OPEN) {
          atlas.send(pcm, { binary: true });
        }
      }

      if (content.generationComplete) {
        sendAtlasJson({ type: "generation_complete" });
      }

      if (content.turnComplete) {
        sendAtlasJson({ type: "turn_complete" });
      }
    }

    if (message.goAway) {
      sendAtlasJson({ type: "go_away", detail: message.goAway });
    }

    if (message.usageMetadata) {
      sendAtlasJson({ type: "usage", usage: message.usageMetadata });
    }
  });

  gemini.on("error", (error) => {
    console.error("Gemini WebSocket error:", error.message);
    sendAtlasJson({ type: "error", source: "gemini", message: error.message });
  });

  gemini.on("close", (code, reason) => {
    console.log(`Gemini closed: ${code} ${reason.toString()}`);
    sendAtlasJson({
      type: "closed",
      source: "gemini",
      code,
      reason: reason.toString(),
    });
    closeBoth(1011, "gemini_closed");
  });

  atlas.on("message", (data, isBinary) => {
    if (!geminiReady || gemini.readyState !== WebSocket.OPEN) {
      return;
    }

    if (isBinary) {
      const audioMessage = {
        realtimeInput: {
          audio: {
            data: Buffer.from(data).toString("base64"),
            mimeType: "audio/pcm;rate=16000",
          },
        },
      };
      gemini.send(JSON.stringify(audioMessage));
      return;
    }

    let control;
    try {
      control = JSON.parse(data.toString());
    } catch {
      return;
    }

    if (control.type === "audio_stream_end") {
      gemini.send(
        JSON.stringify({
          realtimeInput: {
            audioStreamEnd: true,
          },
        }),
      );
    } else if (control.type === "text" && typeof control.text === "string") {
      gemini.send(
        JSON.stringify({
          realtimeInput: {
            text: control.text,
          },
        }),
      );
    } else if (control.type === "ping") {
      sendAtlasJson({ type: "pong", at: Date.now() });
    }
  });

  atlas.on("error", (error) => {
    console.error("Atlas WebSocket error:", error.message);
  });

  atlas.on("close", () => {
    closeBoth(1000, "atlas_disconnected");
  });
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Atlas Live Relay listening on port ${PORT}`);
  console.log(`Model: ${GEMINI_MODEL}`);
  console.log(`Voice: ${ATLAS_VOICE}`);
});
