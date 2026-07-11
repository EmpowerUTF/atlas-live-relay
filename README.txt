ATLAS RELAY v2.1.0 — ON-DEMAND GEMINI SESSION

Replace only server.js in the existing GitHub atlas-live-relay repository.
Do not change Railway variables, domain, device token, model, voice, port, or API key.

Expected health response includes:
"version":"2.1.0-on-demand"
"geminiMode":"on-demand-per-turn"

Expected idle Railway logs stop after startup. Gemini opens only after Atlas submits a /turn request.
