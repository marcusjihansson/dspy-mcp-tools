# Pull Request: Postgres job queue + logs table, secured endpoints, FastAPI worker, client auth

## Summary
Integrates a Postgres-backed job queue with discrete log entries (`job_logs`), normalizes job endpoints, and enables auth/rate-limiting. Adds a FastAPI worker endpoint for async processing and updates the Go client to include Authorization headers.

## Key Changes

### Database
- Switch to PostgreSQL (pgx stdlib) via `DATABASE_URL`.
- Add `job_logs` (append-only, timestamped) with index on `(job_id, id)`.
- `UpdateJobPartial(status, result)` and `AppendJobLog(id, message)`.

### API endpoints (Go)
- `POST /jobs` → returns `{ jobID }` and enqueues work (POST `/process` to FastAPI).
- `PUT /jobs/{id}/update` → worker updates status/result and appends logs.
- `GET /jobs/{id}/status` → returns `{ status, logs: [...] }`.
- `GET /jobs/{id}/result` → returns `202` while running, or `{ result }` once complete.

### Security (Go)
- `SecurityMiddleware` checking `Authorization: Bearer <API_KEY>`.
- Rate limit 5/min per client IP, respects `X-Forwarded-For`.
- Body size limits via `http.MaxBytesReader`.

### FastAPI
- New `POST /process` worker endpoint that calls back to Go with Authorization if `API_KEY` is set.

### Client (Go)
- Reads `API_KEY` from env and adds Authorization header on requests. Added `SetAPIKey` to override programmatically.

### Migrations
- `migrations/001_create_jobs.sql` (jobs, job_logs, index, timestamp trigger).
- `migrations/002_job_logs_and_trigger.sql` (drops legacy `jobs.logs` if exists; idempotent).

### Env vars
- `DATABASE_URL`, `API_KEY`, `FASTAPI_URL`, `GO_SERVER_URL`.

## Testing
1) Apply migrations (psql):
   - `psql "$DATABASE_URL" -f migrations/001_create_jobs.sql`
   - `psql "$DATABASE_URL" -f migrations/002_job_logs_and_trigger.sql`
2) Start FastAPI: `uvicorn fast_server.py:app --port 8000`
3) Start Go: `go run server.go jobqueue.go security.go`
4) Submit job:
   - `curl -X POST http://localhost:8080/jobs -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" -d '{"query":"Regulatory risks for X"}'`
5) Poll:
   - `curl -H "Authorization: Bearer $API_KEY" http://localhost:8080/jobs/<jobID>/status`
   - `curl -H "Authorization: Bearer $API_KEY" http://localhost:8080/jobs/<jobID>/result`

## Breaking changes
- Endpoints changed to `/jobs/{id}/status` and `/jobs/{id}/result`.
- Requires `DATABASE_URL` and (if enforcing auth) `API_KEY`.

---

# Confluence Draft: Postgres Job Queue & Worker Integration – API + Migrations

## Overview
Implements a persistent job queue using PostgreSQL with a separate `job_logs` table for timestamped log entries, secured endpoints with API key auth, and a FastAPI worker for async processing.

## Architecture
Flow:
1. Client → Go (POST `/jobs` { query })
   - Go creates job (`id`, `status=pending`) and returns `jobID`.
   - Go notifies FastAPI worker via POST `/process` { job_id, query }.
2. FastAPI Worker
   - `PUT /jobs/{id}/update` with `status=running` and progressive logs.
   - On finish: `PUT /jobs/{id}/update` with `status=completed`, `result`, and final log.
3. Client Polling
   - `GET /jobs/{id}/status` → `{ status, logs[] }`
   - `GET /jobs/{id}/result` → `202` if running or `{ result }` once complete.

## Endpoints (Go)
- `POST /jobs`
  - Body: `{"query": "Regulatory risks for X"}`
  - Response: `{ "jobID": "<uuid>" }`
- `PUT /jobs/{id}/update`
  - Body: `{ "status": "running|completed|failed", "result": "...", "logs": "text line" }`
  - Behavior: partial updates; logs appended as new entries.
- `GET /jobs/{id}/status`
  - Response: `{ "status": "...", "logs": [{"message": "...", "created_at": "..."}] }`
- `GET /jobs/{id}/result`
  - Response: `202` while running, `200` with `{ "result": "..." }` when completed.

## Worker endpoint (FastAPI)
- `POST /process`
  - Body: `{ "job_id": "<uuid>", "query": "text" }`
  - Behavior: Updates job status/logs via PUT back to Go server; includes Authorization header if `API_KEY` set.

## Database schema (Postgres)
- `jobs(id UUID PK, query TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', result TEXT, created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now())`
- `job_logs(id bigserial PK, job_id UUID FK → jobs(id) ON DELETE CASCADE, message TEXT NOT NULL, created_at timestamptz DEFAULT now())`
- Index: `idx_job_logs_job_id_id(job_id, id)`
- Trigger: `set_timestamp` BEFORE UPDATE ON `jobs` to maintain `updated_at`

## Environment variables
- `DATABASE_URL=postgres://user:pass@host:5432/db?sslmode=disable`
- `API_KEY=your_server_api_key`
- `FASTAPI_URL=http://localhost:8000`
- `GO_SERVER_URL=http://localhost:8080`

## Security
- Authorization: `Bearer <API_KEY>` on all requests to Go server.
- Rate limiting: 5 requests/min per client IP.

