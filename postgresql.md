# PostgreSQL Setup and Usage Guide

This guide covers PostgreSQL 17 setup with pgvector extension for AI/ML applications.

## Installation and Service Management

### Starting and Stopping PostgreSQL

```bash
# Start PostgreSQL 17 service
brew services start postgresql@17

# Stop PostgreSQL 17 service
brew services stop postgresql@17

# Restart PostgreSQL 17 service
brew services restart postgresql@17

# Check which PostgreSQL services are running
brew services list | grep postgresql
```

## Database Management

### Creating a Database

```bash
# Create a new database
createdb ai_law_db

# Create database with specific owner
createdb -O username database_name

# List all databases
psql -l
```

### Connecting to Database

```bash
# Connect to database (simple)
psql ai_law_db

# Connect with specific parameters
psql -h localhost -p 5432 -U name -d ai_law_db

# Connect and run a single command
psql ai_law_db -c "SELECT version();"
```

## Vector Extension Setup

### Installing pgvector Extension

```bash
# Enable vector extension in your database
psql ai_law_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify vector extension is installed
psql ai_law_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Working with Vector Data

```sql
-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1024),  -- Adjust dimension as needed
    metadata JSONB
);

-- Create HNSW index for fast similarity search
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- Insert vector data
INSERT INTO documents (content, embedding)
VALUES ('sample text', '[0.1, 0.2, 0.3, ...]');

-- Similarity search
SELECT content, embedding <=> '[0.1, 0.2, 0.3, ...]' AS distance
FROM documents
ORDER BY distance
LIMIT 5;
```

## Common Database Operations

### Viewing Database Structure

```sql
-- List all tables
\dt

-- Describe table structure
\d table_name

-- List all indexes
\di

-- Show table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public';
```

### Data Inspection

```sql
-- Count rows in table
SELECT COUNT(*) FROM table_name;

-- View sample data
SELECT * FROM table_name LIMIT 10;

-- Check for null values
SELECT COUNT(*) FROM table_name WHERE column_name IS NULL;
```

### Maintenance Commands

```sql
-- Update table statistics
ANALYZE table_name;

-- Rebuild indexes
REINDEX TABLE table_name;

-- Vacuum table (cleanup)
VACUUM table_name;
```

## Useful psql Commands

```bash
# Inside psql prompt:
\l          # List databases
\c dbname   # Connect to database
\dt         # List tables
\d table    # Describe table
\q          # Quit psql
\?          # Help with psql commands
\h          # Help with SQL commands

# Execute SQL file
\i /path/to/file.sql

# Output query results to file
\o output.txt
SELECT * FROM table_name;
\o  # Stop output to file
```

## Environment Variables

Set these in your `.env` file for application connections:

```bash
DB_NAME="ai_law_db"
DB_USER="your_username"
DB_PASSWORD="your_password"
DB_HOST="localhost"
DB_PORT="5432"
```

## Troubleshooting

### Common Issues

```bash
# Check if PostgreSQL is running
brew services list | grep postgresql

# Check PostgreSQL logs
tail -f /opt/homebrew/var/log/postgresql@17.log

# Test connection
pg_isready -h localhost -p 5432

# Reset user password (if needed)
psql postgres -c "ALTER USER username PASSWORD 'newpassword';"
```

### Performance Monitoring

```sql
-- Check active connections
SELECT * FROM pg_stat_activity;

-- Check database size
SELECT pg_database_size('ai_law_db');

-- Check slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

# Test if data has been inserted:

-- Check total documents  
 SELECT COUNT(\*) FROM rag_documents;

-- View documents by file  
 SELECT  
 metadata->>'filename' as filename,  
 COUNT(\*) as chunks  
 FROM rag_documents  
 GROUP BY metadata->>'filename'  
 ORDER BY chunks DESC;

-- Search content  
 SELECT  
 LEFT(content, 100) as preview,  
 metadata->>'filename' as file  
 FROM rag_documents  
 WHERE content ILIKE '%your search term%'  
 LIMIT 5;

-- Check database size  
 SELECT pg_size_pretty(pg_database_size('ai_law_db')) as database_size;
