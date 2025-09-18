// jobqueue.go
package main

import (
	"context"
	"database/sql"
	"errors"
	"os"
	"time"

	"github.com/google/uuid"
	_ "github.com/jackc/pgx/v5/stdlib"
)

type LogEntry struct {
	Message   string    `json:"message"`
	CreatedAt time.Time `json:"created_at"`
}

type Job struct {
	ID        string     `json:"id"`
	Query     string     `json:"query"`
	Status    string     `json:"status"`
	Result    string     `json:"result,omitempty"`
	Logs      []LogEntry `json:"logs,omitempty"`
	CreatedAt time.Time  `json:"created_at"`
	UpdatedAt time.Time  `json:"updated_at"`
}

var db *sql.DB

func InitDB() error {
	var err error
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		return errors.New("DATABASE_URL is not set")
	}
	// Use pgx stdlib driver
	db, err = sql.Open("pgx", dsn)
	if err != nil {
		return err
	}
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(30 * time.Minute)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		return err
	}

	// Create tables if not exist (bootstrap; migrations will handle in real env)
	schema := `
	CREATE TABLE IF NOT EXISTS jobs (
		id UUID PRIMARY KEY,
		query TEXT NOT NULL,
		status TEXT NOT NULL DEFAULT 'pending',
		result TEXT,
		created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
		updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
	);
	CREATE TABLE IF NOT EXISTS job_logs (
		id BIGSERIAL PRIMARY KEY,
		job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
		message TEXT NOT NULL,
		created_at TIMESTAMPTZ NOT NULL DEFAULT now()
	);
	`
	_, err = db.ExecContext(ctx, schema)
	return err
}

func CreateJob(query string) (Job, error) {
	id := uuid.New().String()
	job := Job{
		ID:     id,
		Query:  query,
		Status: "pending",
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	// created_at/updated_at default in DB
	_, err := db.ExecContext(ctx, `INSERT INTO jobs (id, query, status) VALUES ($1, $2, $3)`, job.ID, job.Query, job.Status)
	if err != nil {
		return Job{}, err
	}
	// Fetch timestamps
	return GetJob(job.ID)
}

// UpdateJobPartial updates provided fields (nil means no change)
func UpdateJobPartial(id string, status, result *string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if status == nil && result == nil {
		// only touch updated_at
		_, err := db.ExecContext(ctx, `UPDATE jobs SET updated_at = now() WHERE id = $1`, id)
		return err
	}
	if status != nil && result != nil {
		_, err := db.ExecContext(ctx, `UPDATE jobs SET status = $1, result = $2, updated_at = now() WHERE id = $3`, *status, *result, id)
		return err
	}
	if status != nil {
		_, err := db.ExecContext(ctx, `UPDATE jobs SET status = $1, updated_at = now() WHERE id = $2`, *status, id)
		return err
	}
	_, err := db.ExecContext(ctx, `UPDATE jobs SET result = $1, updated_at = now() WHERE id = $2`, *result, id)
	return err
}

func AppendJobLog(id, message string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := db.ExecContext(ctx, `INSERT INTO job_logs (job_id, message) VALUES ($1, $2)`, id, message)
	return err
}

func GetJob(id string) (Job, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var job Job
	row := db.QueryRowContext(ctx, `SELECT id, query, status, COALESCE(result, ''), created_at, updated_at FROM jobs WHERE id = $1`, id)
	if err := row.Scan(&job.ID, &job.Query, &job.Status, &job.Result, &job.CreatedAt, &job.UpdatedAt); err != nil {
		return Job{}, err
	}
	// Load logs
	rows, err := db.QueryContext(ctx, `SELECT message, created_at FROM job_logs WHERE job_id = $1 ORDER BY id ASC`, id)
	if err != nil {
		return Job{}, err
	}
	defer rows.Close()
	logs := make([]LogEntry, 0, 8)
	for rows.Next() {
		var le LogEntry
		if err := rows.Scan(&le.Message, &le.CreatedAt); err != nil {
			return Job{}, err
		}
		logs = append(logs, le)
	}
	job.Logs = logs
	return job, rows.Err()
}
