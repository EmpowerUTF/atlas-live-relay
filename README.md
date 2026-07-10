# Atlas Live Relay

Low-latency WebSocket relay between the Atlas ESP32 and Gemini Live.

## Required Railway variables

- `GEMINI_API_KEY` — the Gemini key created in Google AI Studio
- `ATLAS_DEVICE_TOKEN` — a private random password used only by Atlas to connect

## Optional Railway variables

- `GEMINI_MODEL` — defaults to `gemini-3.1-flash-live-preview`
- `ATLAS_VOICE` — defaults to `Kore`
- `ATLAS_SYSTEM_INSTRUCTION` — Atlas personality/instructions

## Routes

- `GET /health` — health check
- `WSS /atlas?token=YOUR_DEVICE_TOKEN` — Atlas audio WebSocket

## Protocol

Atlas to relay:

- Binary frames: raw mono PCM, 16-bit little-endian, 16 kHz
- Text JSON `{ "type": "audio_stream_end" }`
- Text JSON `{ "type": "ping" }`

Relay to Atlas:

- Binary frames: raw mono PCM, 16-bit little-endian, 24 kHz
- Text JSON status messages such as `ready`, `input_transcript`, `output_transcript`, `turn_complete`, and `error`
