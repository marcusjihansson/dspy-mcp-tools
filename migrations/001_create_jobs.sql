-- Jobs base table (PostgreSQL)
-- Note: the application (Go) generates UUIDs; no default UUID generator is defined here.

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    query TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Separate logs table to store step-by-step logs with timestamps
CREATE TABLE IF NOT EXISTS job_logs (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Ensure efficient retrieval of logs per job in chronological order
CREATE INDEX IF NOT EXISTS idx_job_logs_job_id_id ON job_logs (job_id, id);

-- Trigger to automatically maintain updated_at timestamp on updates
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
