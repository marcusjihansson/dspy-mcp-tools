# MCP Financial Analysis Platform

A dual-language (Python + Go) platform that exposes a suite of financial analysis tools via an HTTP API and CLIs. It combines a Python FastAPI service that implements the financial tools and the MCP-compliant interfaces with a Go API gateway that adds robust session management, proxying, and parallel/sequential execution orchestration.

The platform supports:

- Asset price analysis (stocks and crypto via Alpaca)
- News and market sentiment analysis
- Company fundamentals
- Macroeconomic context
- Company profile generation
- Regulatory/RAG analysis (powered by PostgreSQL + local embeddings)

This README consolidates content from server.md, client_cli.md, config.md, and postgresql.md into a single quick-start and reference guide.

## High-level architecture

```
+------------------------+       HTTP (JSON)       +---------------------------+
|  Python FastAPI (8000) |<----------------------->|  Go API Server (8080)     |
|  fast_server.py        |                         |  server.go                |
|  - Implements MCP      |                         |  - Session management     |
|    /mcp/tools, /mcp/call|                        |  - Proxies to FastAPI     |
|  - Tools: price,        |                         |  - Query analyze routes   |
|    sentiment,           |                         |  - Parallel/sequential    |
|    fundamentals, macro, |                         |    execution              |
|    profile, regulatory  |                         |                           |
+------------^-----------+                         +-------------^-------------+
             |                                                         |
             |  Calls tools implemented in Python classes              |
             |  (asset_price_tool.py, sentiment_tool.py, etc.)        |
             v                                                         |
  +----------------------+                                          Clients
  |  DSPy-based tools    |                                          (HTTP/CLI)
  |  + Alpaca API        |
  |  + PostgreSQL RAG    |
  |  + Ollama/OpenRouter |
  +----------------------+
```

- Most tool logic lives in Python (see asset_price_tool.py, fundamental_tool.py, macro_tool.py, company_profile.py, sentiment_tool.py, regulatory.py).
- The Python service is the MCP Tool Server and also supports auto-discovery and bulk execution.
- The Go service provides a stable HTTP surface, session management, and proxies to the Python service. It exposes mirrored MCP endpoints and higher-level analysis routes.

## Features

- MCP-compliant tool listing and execution
- Parallel and sequential multi-tool execution
- Natural language query analysis with auto tool selection
- Go and Python client libraries and CLIs
- Pluggable LLM backends (OpenRouter, Ollama; see config.py)
- Regulatory analysis backed by a local PostgreSQL RAG store

## Repository layout

- fast_server.py — Python FastAPI server exposing tools and MCP endpoints
- server.go — Go API server proxy and session manager
- client.py — Python client library
- client.go — Go client library
- cli.py — Python CLI
- cli.go — Go CLI
- asset_price_tool.py, sentiment_tool.py, fundamental_tool.py, macro_tool.py, company_profile.py, regulatory.py — Tool implementations
- config.py, log.py, utility.py — Shared configuration, logging, and utilities
- server.md, client_cli.md, config.md, postgresql.md — In-depth documentation
- AI_law/ — Regulatory PDFs used for RAG ingestion (see postgresql.md)

## Prerequisites

- Python 3.10+
- Go 1.20+
- PostgreSQL (for regulatory analyzer)
- Optional: Ollama running locally if you choose Ollama models
- Optional: OpenRouter API key if you choose OpenRouter models
- Alpaca API key/secret for price data

## Python setup

1. Create and activate a virtual environment

- macOS/Linux
  - python3 -m venv .venv
  - source .venv/bin/activate
- Windows (PowerShell)
  - py -3 -m venv .venv
  - .venv\\Scripts\\Activate.ps1

2. Install dependencies

The project uses the following Python packages:

- fastapi, uvicorn
- pydantic
- dspy-ai (or dspy)
- python-dotenv
- requests
- pandas, numpy
- alpaca-py
- psycopg2 (for PostgreSQL)
- ollama (Python client)

Install with pip:

pip install fastapi uvicorn pydantic dspy-ai python-dotenv requests pandas numpy alpaca-py psycopg2-binary ollama

Note: If your environment requires it, you can use psycopg2-binary for convenience during development.

3. Environment variables (.env)

Create a .env in the repo root with the following keys as needed:

# LLM config (example using OpenRouter)

CHATBOT_KEY=your_openrouter_key

# Alpaca API for price data

ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret

# PostgreSQL for regulatory analyzer

DB_NAME=ai_law_db
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432

# Optional Ollama and/or NIM settings are configured via config.py

4. Start the Python server (port 8000)

python fast_server.py

This runs uvicorn with app on 0.0.0.0:8000.

## Go setup

1. Ensure Go is installed (1.20+). Initialize modules if needed (go mod init ... if not already present).

2. Environment variables for Go server

- PORT=8080 (default)
- PYTHON_HOST=localhost
- PYTHON_PORT=8000
- TIMEOUT_SECONDS=30

3. Run the Go server (port 8080)

go run server.go

The Go server expects the Python server to be reachable at PYTHON_HOST:PYTHON_PORT.

## Quick start (two terminals)

Terminal 1 (Python FastAPI):

- source .venv/bin/activate
- python fast_server.py

Terminal 2 (Go API server):

- go run server.go

Sanity check:

- curl http://localhost:8080/health
- curl http://localhost:8080/info
- curl http://localhost:8080/mcp/tools

## Using the CLIs

Python CLI

- python cli.py health
- python cli.py server_info
- python cli.py tools
- python cli.py analyze_price AAPL 2024-01-01 2024-06-01
- python cli.py analyze_sentiment AAPL 2024-01-01 2024-06-01
- python cli.py analyze_fundamentals AAPL 4
- python cli.py analyze_macro AAPL
- python cli.py company_profile AAPL
- python cli.py regulatory_compliance NVDA
- python cli.py comprehensive AAPL 2024-01-01 2024-06-01
- python cli.py due_diligence AAPL
- python cli.py sentiment_analysis AAPL 2024-01-01 2024-06-01

Go CLI

- go run cli.go health
- go run cli.go server_info
- go run cli.go tools
- go run cli.go analyze_price AAPL 2024-01-01 2024-06-01
- go run cli.go analyze_sentiment AAPL 2024-01-01 2024-06-01
- go run cli.go analyze_fundamentals AAPL 4
- go run cli.go analyze_macro AAPL
- go run cli.go company_profile AAPL
- go run cli.go regulatory_compliance NVDA
- go run cli.go comprehensive AAPL 2024-01-01 2024-06-01
- go run cli.go due_diligence AAPL
- go run cli.go sentiment_analysis AAPL 2024-01-01 2024-06-01

## Key endpoints (Go server, default port 8080)

- GET /health — health of Go + Python services
- GET /info — server name/version/capabilities
- GET /examples — API usage examples
- GET /mcp/tools — MCP tool list (name, description, JSON schema)
- POST /mcp/call — Call a single MCP tool
  Body: { "name": "analyze_price", "arguments": {"symbol": "AAPL", "start_date": "2024-01-01", "end_date": "2024-06-01"} }
- POST /query/analyze — Smart query analysis with detailed report
  Body: { "query": "comprehensive analysis of AAPL", "execution_mode": "parallel" }
- POST /api/analyze — Legacy analyze API returning a compact result
  Body: { "query": "comprehensive analysis of AAPL", "execution_mode": "parallel" }
- GET /api/tools/discover — Discover tools (via Python FastAPI)
- POST /tools/execute — Execute multiple tools (parallel or sequential)

Note: The Python FastAPI also exposes its native paths (/mcp/tools, /mcp/call, /tools/execute, /query/analyze, /health), but the Go server is the recommended entry point.

## Configuration and models

- config.py centralizes LLM provider configuration and logging defaults.
- Set OpenRouter (CHATBOT_KEY) or Ollama models using helper methods in Config.
- log.py configures structured, rotating logs.
- utility.py provides caching and retry decorators.

If Alpaca credentials are not present, AssetPriceAnalyzer will raise a clear error. If PostgreSQL isn’t configured or populated, the RegulatoryAnalyzer will fail; see next section for setup.

## Regulatory analyzer and PostgreSQL RAG

- See postgresql.md for end-to-end instructions.
- Summary:
  - Create a database (e.g., ai_law_db) and table rag_documents with a vector column for embeddings.
  - Ingest the PDFs from AI_law/ into the database.
  - Ensure Ollama is running and the bge-m3:latest embedding model is available.
  - Configure DB\_\* environment vars in .env.

Once populated, the RegulatoryAnalyzer performs a two-stage DSPy pipeline:

1. Retrieve and summarize regulatory fragments relevant to a question.
2. Apply them to a specific ticker’s business context (impact and risk).

## Notes on launch script

- launch_servers.zsh currently contains hard-coded conda env and path examples for one machine. Prefer running the two servers manually (see Quick start), or adapt the script to your environment.

## Development tips and gotchas

- Start Python first, then Go (Go proxies to Python).
- Use X-Session-ID header (both clients set a session by default) to keep request context.
- If you encounter missing packages, install the Python dependencies listed above.
- For Alpaca API, ensure your symbol is supported; crypto symbols may require "/USD" pairs.
- On Windows, use psycopg2-binary for easier local setup.
- If using Ollama, run `ollama serve` and confirm models with `ollama list`.

## Roadmap / suggestions

- Add requirements.txt and go.mod/go.sum for reproducible installs.
- Provide Dockerfiles and docker-compose for Python, Go, and Postgres.
- Add unit/integration tests for tools and endpoints.
- Harden input validation across endpoints.
- Replace hard-coded paths in launch_servers.zsh.
- Document environment variable matrix in a single place.

## Disclaimer

This project is for educational and research use. Financial outputs are not investment advice. Verify any insights independently.

## License

Feel free to use it as you like. Read the disclaimer, other than that feel free to pull this and then go at it.

But if you like what I have done, can you reach me on:

LinkedIn: https://www.linkedin.com/in/marcus-frid-johansson/

Twitter/x: https://x.com/marcusjihansson
