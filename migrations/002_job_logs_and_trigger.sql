-- Migration 002: Separate logs table and timestamp trigger

-- If the initial migration has already created a 'logs' column on jobs, drop it
ALTER TABLE IF EXISTS jobs DROP COLUMN IF EXISTS logs;

-- Create job_logs table to store discrete log entries with timestamps
CREATE TABLE IF NOT EXISTS job_logs (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index for efficient chronological retrieval
CREATE INDEX IF NOT EXISTS idx_job_logs_job_id_id ON job_logs (job_id, id);

-- Ensure updated_at is maintained automatically on jobs updates
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
