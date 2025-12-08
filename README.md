# DiligenceSquared Take-Home

This repo scaffolds a unified flow that sends respondents through a Typeform screener and then directly into an ElevenLabs conversational interview with stateful resume, progress tracking, and transcript storage.

## High-level architecture
- `web/`: lightweight single-page UI (no build step) that embeds the Typeform with hidden respondent IDs, persists session locally, and hands off to the voice interview panel.
- `server/`: FastAPI backend that tracks respondent state, receives Typeform webhooks, orchestrates ElevenLabs conversations, and stores transcripts/snapshots on disk.
- `data/`: JSON + transcript files (gitignored) to keep sessions resumable without a database.
- `docs/`: notes on assumptions and next steps.

## Quick start (localhost)
1) Create `.env` from `.env.example` and fill in Typeform + ElevenLabs keys.
2) Create the webhook in Typeform pointing to `http://localhost:8000/api/typeform/webhook` with the shared secret.
3) Install backend deps: `cd server && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
4) Run API: `uvicorn main:app --reload`.
5) Serve static UI (any static server): `python -m http.server 3000 --directory web`.
6) Visit `http://localhost:3000`, start a session, complete the form, then launch the interview.

## Flow overview
1) UI calls `POST /api/session` to allocate/resume a `respondent_id` and stores it in `localStorage`.
2) The Typeform embed receives the `respondent_id` as a hidden field so every response carries it.
3) Typeform webhook hits `POST /api/typeform/webhook` with the full response; backend validates the secret, extracts segmentation (BMW Customer vs Potential), and stores it with the session.
4) UI polls `GET /api/session/{id}` to detect when screening is qualified, then enables the “Start interview” button.
5) UI calls `POST /api/conversation/start` to open (or resume) an ElevenLabs conversation tied to `respondent_id` and the assigned segment. Conversation IDs are persisted for resume.
6) UI streams audio/chat with ElevenLabs (placeholder hook in `web/index.html`) and shows progress. On completion, `POST /api/conversation/complete` persists transcript and status.

## What’s included vs. TODO
- Included: skeleton API endpoints, Typeform webhook handler, file-backed session store, static UI shell with Typeform embed + interview pane, and notes on resumption strategy.
- TODO: wire the real ElevenLabs streaming client in `web/index.html`, tighten auth (JWT/cookies), add tests, and deploy scripts (Fly/Render/ngrok).

See `docs/notes.md` for implementation details and a task backlog.
