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
        version: "1.3.0-manual-vad",
        model: GEMINI_MODEL,
        voice: ATLAS_VOICE,
      }),
    );
    return;
  }

  res.writeHead(404, { "content-type": "application/json" });
  res.end(JSON.stringify({ ok: false, error: "not_found" }));
});

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

  const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

  let session = null;
  let geminiReady = false;
  let closing = false;
  let connectInFlight = false;
  let reconnectTimer = null;
  let inputAudioBytes = 0;
  let outputAudioBytes = 0;

  const sendAtlasJson = (payload) => {
    if (atlas.readyState === WebSocket.OPEN) {
      atlas.send(JSON.stringify(payload));
    }
  };

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

        if (atlas.readyState === WebSocket.OPEN) {
          atlas.send(pcm, { binary: true });
        }
      }

      if (content.generationComplete) {
        console.log(`Gemini generation complete. Output bytes: ${outputAudioBytes}`);
        sendAtlasJson({ type: "generation_complete" });
      }

      if (content.turnComplete) {
        console.log(`Gemini turn complete. Output bytes: ${outputAudioBytes}`);
        sendAtlasJson({ type: "turn_complete" });
        outputAudioBytes = 0;
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

  atlas.on("close", () => {
    console.log("Atlas device disconnected.");
    closing = true;
    geminiReady = false;

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

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Atlas Live Relay listening on port ${PORT}`);
  console.log("Relay version: 1.3.0-manual-vad");
  console.log(`Model: ${GEMINI_MODEL}`);
  console.log(`Voice: ${ATLAS_VOICE}`);
});
