## Architecture + decisions
- **State storage:** File-backed JSON (`data/sessions.json`) plus per-session transcript files in `data/transcripts/{respondent_id}.json`. Good enough for take-home; swap for Redis/Postgres later.
- **Identity:** The frontend requests a `respondent_id` from `/api/session` and passes it to Typeform as a hidden field. Same ID keys the ElevenLabs conversation. LocalStorage mirrors it so users can leave/return.
- **Typeform resume:** Typeform handles in-form resume when hidden params match. We also persist answers server-side after webhook reception to allow re-hydration if needed.
- **Qualification:** Webhook extracts brand ownership and labels `segment` as `customer` (BMW) or `potential` (Mercedes/Audi). Others terminate (`screened_out`).
- **Interview routing:** Segment selects the question set (customer vs potential). Progress is tracked server-side; frontend shows a progress bar and disables completion until required items are answered.
- **Conversation resumption:** Conversation ID and last asked index stored in session. On reconnect, frontend requests `/api/conversation/start` again; backend returns existing conversation_id and history so the client can prompt ElevenLabs with context.
- **Transcripts:** `/api/conversation/complete` stores the transcript and marks status `interview_complete`.

## API sketch
- `POST /api/session` → `{respondent_id, state}`
- `GET /api/session/{id}` → session payload (segment, status, conversation_id, progress)
- `POST /api/typeform/webhook` (validated via secret) → updates segment/status/responses
- `POST /api/conversation/start` → returns `conversation_id`, `segment`, `questions`, `history`
- `POST /api/conversation/complete` → persist transcript and status

## TODO backlog
- Hook ElevenLabs Conversational AI streaming client in the web UI (mic + audio playback).
- Add automated tests for the storage and webhook parsing.
- Add CSRF/auth for the admin endpoints; rate limit webhook.
- Add UI polish: progress indicator, reconnection banner, transcript download.
- Deployment: containerize + provide ngrok script for webhook testing.
