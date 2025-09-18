# Database Migration Guide

This guide documents the job orchestration schema for PostgreSQL and the steps to apply or update your database.

## Overview

We migrated from a single `jobs` table (with a single `logs` TEXT column) to a normalized model using two tables:

- `jobs` — Stores the job metadata and current status/result
- `job_logs` — Stores append-only, timestamped log entries for each job

Additionally, we added a trigger to automatically maintain `updated_at` on `jobs` whenever a row is updated.

## Schema

```sql
-- jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    query TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- job_logs table
CREATE TABLE IF NOT EXISTS job_logs (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Helpful index for chronological retrieval of logs per job
CREATE INDEX IF NOT EXISTS idx_job_logs_job_id_id ON job_logs (job_id, id);

-- Trigger to maintain updated_at on jobs
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_timestamp ON jobs;
CREATE TRIGGER set_timestamp
BEFORE UPDATE ON jobs
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();
```

## Files

- `migrations/001_create_jobs.sql`
  - Creates `jobs`
  - Creates `job_logs`
  - Creates index `idx_job_logs_job_id_id`
  - Creates `set_timestamp` trigger
- `migrations/002_job_logs_and_trigger.sql`
  - Safe, incremental migration:
    - Drops legacy `jobs.logs` column if present
    - Re-creates `job_logs`, index, and `set_timestamp` trigger

## How to apply

1. Set the `DATABASE_URL` environment variable (used by the Go server):

```bash
export DATABASE_URL="postgres://postgres:postgres@localhost:5432/ai_law_db?sslmode=disable"
```

2. Apply the migrations with `psql` (or your own migration tool):

```bash
psql "$DATABASE_URL" -f migrations/001_create_jobs.sql
psql "$DATABASE_URL" -f migrations/002_job_logs_and_trigger.sql
```

If you are starting from an empty database, running `001` and then `002` is safe and idempotent.

## Application integration

- The Go server initializes the DB via `InitDB()` and uses `pgx` (stdlib) with `DATABASE_URL`.
- `UpdateJobPartial(id, status, result)` supports partial field updates.
- `AppendJobLog(id, message)` inserts a new row into `job_logs`.
- `GetJob(id)` returns the `jobs` row and an ordered array of structured log entries.

## Environment variables

Update your `.env` (or config) accordingly:

```env
# DB connection for Go server
DATABASE_URL=postgres://postgres:postgres@localhost:5432/ai_law_db?sslmode=disable

# Security
API_KEY=your_server_api_key

# Service URLs
FASTAPI_URL=http://localhost:8000
GO_SERVER_URL=http://localhost:8080
```

## Notes

- Keep migration files in version control. They document your schema history.
- The server will create tables if they do not exist, but migrations should remain the source of truth for deployments.
- Consider adding a migration runner (e.g., `golang-migrate/migrate`) for production workflows.
