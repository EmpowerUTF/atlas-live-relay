ATLAS LIVE RELAY v2.2 — BUFFERED RESPONSE

Replace only server.js in the existing GitHub relay repository.
Do not change package.json, Railway variables, keys, token, domain, port, model, or voice.

Health must show:
  "version":"2.2.0-buffered-response"
  "responseMode":"buffered-content-length"

This version buffers the complete Gemini PCM response on Railway, then sends one fixed-length HTTP body to Atlas.
