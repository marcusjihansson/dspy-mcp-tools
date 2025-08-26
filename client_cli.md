# Financial Analysis Clients & CLIs Documentation

This document provides comprehensive information about the **Python** and **Go** client libraries and command-line interfaces (CLIs) for interacting with the Financial Analysis Server Architecture.

---

## Architecture Overview

The client libraries and CLI tools act as convenient interfaces to the Go Server (port `8080`), which proxies requests to the Python FastAPI server (port `8000`).

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Python CLI    │    │    Go CLI       │    │  Client Libs    │
│   (cli.py)      │    │   (cli.go)      │    │ (client.py/go)  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼───────────────┐
                    │      Go Server              │
                    │    (localhost:8080)         │
                    │  - Request routing          │
                    │  - Session management       │
                    │  - Error handling           │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   Python FastAPI Server    │
                    │    (localhost:8000)         │
                    │  - Financial analysis       │
                    │  - MCP tool execution       │
                    │  - Data processing          │
                    └─────────────────────────────┘
```

### Key Features

- **Unified Interface**: Both Python and Go implementations provide identical functionality
- **Session Management**: Automatic session tracking via `X-Session-ID` headers
- **Error Handling**: Comprehensive error handling with meaningful messages
- **Request Formatting**: Automatic JSON serialization/deserialization
- **Timeout Management**: Configurable request timeouts (default: 120 seconds)
- **Connection Pooling**: Efficient HTTP connection reuse

---

## Setup & Installation

### Python Client & CLI

#### Prerequisites

- Python 3.9 or higher
- `requests` library for HTTP operations

#### Installation

```bash
# Install dependencies
pip install requests

# Verify installation
python -c "import requests; print('✓ Dependencies installed')"
```

#### Usage

```bash
# Direct CLI execution
python cli.py health
python cli.py analyze_price AAPL 2024-01-01 2024-08-01

# Using as a library
python -c "
from client import FinancialAnalysisClient
client = FinancialAnalysisClient()
print(client.health_check())
"
```

### Go Client & CLI

#### Prerequisites

- Go 1.19 or higher
- Standard library packages (no external dependencies)

#### Installation

```bash
# Build CLI binary
go build -o fincli cli.go

# Verify build
./fincli health

# Alternative: Run without building
go run cli.go health
```

#### Usage

```bash
# Using compiled binary
./fincli health
./fincli analyze_price AAPL 2024-01-01 2024-08-01

# Direct execution
go run cli.go analyze_price TSLA 2024-01-01 2024-12-31
```

### Environment Configuration

Both clients support configuration via environment variables:

```bash
# Optional: Set custom server URL
export FINANCIAL_SERVER_URL="http://localhost:8080"

# Optional: Set session ID
export SESSION_ID="my_custom_session"
```

---

## Available Commands

Both the Python and Go CLIs support the same set of commands.

### Health & Info

- `health` — Check server health
- `server_info` — Get server information

### Analysis Tools

- `analyze_price SYMBOL START_DATE END_DATE` — Includes technical indicators automatically
- `analyze_sentiment SYMBOL START_DATE END_DATE`
- `analyze_fundamentals SYMBOL QUARTERS`
- `analyze_macro SYMBOL`
- `company_profile SYMBOL`
- `regulatory_compliance SYMBOL`

### Composite Analyses

- `comprehensive SYMBOL START_DATE END_DATE` — Comprehensive analysis
- `due_diligence SYMBOL` — Sequential due diligence analysis
- `sentiment_analysis SYMBOL START_DATE END_DATE` — Sentiment-focused analysis

### Tool Management

- `tools` — List all MCP tools
- `discover_tools` — Discover available tools
- `manual SYMBOL` — Execute multiple tools manually

### Utility / Examples

- `health_check` — Example: health check + server info
- `list_tools` — Example: print available tools
- `date_utilities` — Example: format current date/time

---

## Command to API Endpoint Mapping

| CLI Command             | HTTP Method | Endpoint                        | Notes                               |
| ----------------------- | ----------- | ------------------------------- | ----------------------------------- |
| `health`                | GET         | `/health`                       | Server health status                |
| `server_info`           | GET         | `/info`                         | Server info and capabilities        |
| `analyze_price`         | POST        | `/mcp/call`                     | Tool: analyze_price                 |
| `analyze_sentiment`     | POST        | `/mcp/call`                     | Tool: analyze_sentiment             |
| `analyze_fundamentals`  | POST        | `/mcp/call`                     | Tool: analyze_fundamentals          |
| `analyze_macro`         | POST        | `/mcp/call`                     | Tool: analyze_macro                 |
| `company_profile`       | POST        | `/mcp/call`                     | Tool: generate_company_profile      |
| `regulatory_compliance` | POST        | `/mcp/call`                     | Tool: analyze_regulatory_compliance |
| `comprehensive`         | POST        | `/mcp/call` or `/query/analyze` | Smart query analysis                |
| `due_diligence`         | POST        | `/mcp/call` or `/query/analyze` | Sequential due diligence            |
| `sentiment_analysis`    | POST        | `/mcp/call` or `/query/analyze` | Sentiment & news                    |
| `tools`                 | GET         | `/mcp/tools`                    | List MCP tools                      |
| `discover_tools`        | GET         | `/api/tools/discover`           | Tool discovery                      |
| `manual`                | POST        | `/tools/execute`                | Multi-tool execution                |
| `health_check`          | GET/GET     | `/health`, `/info`              | Example                             |
| `list_tools`            | GET         | `/mcp/tools`                    | Example                             |
| `date_utilities`        | —           | Local function                  | Date formatting utility             |

---

## Example Usage

### Python CLI

```bash
python cli.py health
python cli.py analyze_price AAPL 2024-01-01 2024-12-31
python cli.py comprehensive MSFT 2024-06-01 2024-12-31
```

### Go CLI

```bash
./fincli health
./fincli analyze_price TSLA 2024-01-01 2024-12-31
./fincli sentiment_analysis AAPL 2024-05-01 2024-07-01
```

---

## How It Works

1. **CLI / Client Library** — Reads command and arguments from terminal or function call
2. **HTTP Request** — Builds appropriate JSON payload and sends it to the Go server at `http://localhost:8080`
3. **Go Server** — Processes the request, applies session/error handling, and forwards to Python FastAPI server if needed
4. **Python FastAPI Server** — Executes the actual financial analysis logic and returns JSON
5. **CLI Output** — Parses and prints the server's JSON response

---

## Session Management

Both Python and Go clients automatically attach an `X-Session-ID` header, which can be set manually or auto-generated.

---

## Client Library Implementation Details

### Python Client (`client.py`)

The `FinancialAnalysisClient` class provides a comprehensive Python interface:

#### Core Features

- **Session Management**: Automatic session creation with unique IDs
- **Context Manager Support**: Use with `with` statements for automatic cleanup
- **Error Handling**: Custom `FinancialAnalysisClientError` exceptions
- **Request Pooling**: Reuses HTTP connections for efficiency

#### Key Methods

```python
# Initialize client
client = FinancialAnalysisClient(
    base_url="http://localhost:8080",
    timeout=120,
    session_id="optional_custom_id"
)

# Health & Info
client.health_check()                    # Full health response
client.is_healthy()                      # Boolean health check
client.get_server_info()                 # Server capabilities
client.get_examples()                    # API usage examples

# MCP Tool Interface
client.list_mcp_tools()                  # Available tools
client.call_mcp_tool(name, arguments)    # Direct tool execution

# Financial Analysis
client.analyze_price(symbol, start, end, include_technical=True)
client.analyze_sentiment(symbol, start, end, sources=[])
client.analyze_fundamentals(symbol, quarters=4, include_*)
client.analyze_macro(symbol, include_sector=True, include_economic=True)
client.generate_company_profile(symbol)
client.analyze_regulatory_compliance(symbol, include_ai=True)

# Smart Query Analysis
client.query_analyze(query, execution_mode="parallel")
client.api_analyze(query, execution_mode="parallel")

# Multi-Tool Execution
client.execute_tools(tools_list, parameters, execution_mode="parallel")
client.discover_tools()

# Convenience Methods
client.comprehensive_analysis(symbol, start, end)
client.due_diligence_analysis(symbol)
client.sentiment_analysis(symbol, start, end)
```

#### Usage Patterns

```python
# Context manager (recommended)
with FinancialAnalysisClient() as client:
    result = client.analyze_price("AAPL", "2024-01-01", "2024-12-31")
    print(result)

# Manual session management
client = FinancialAnalysisClient()
try:
    result = client.comprehensive_analysis("MSFT", "2024-01-01", "2024-12-31")
finally:
    client.close()

# Error handling
try:
    result = client.analyze_fundamentals("INVALID")
except FinancialAnalysisClientError as e:
    print(f"Analysis failed: {e}")
```

### Go Client (`client.go`)

The Go client provides equivalent functionality with Go idioms:

#### Core Features

- **Type Safety**: Strongly typed method signatures
- **Error Handling**: Standard Go error patterns
- **HTTP Client**: Configurable timeout and connection pooling
- **JSON Marshaling**: Automatic request/response serialization

#### Key Methods

```go
// Initialize client
client := NewClient("http://localhost:8080", "session_id", 120)

// Health & Info
health, err := client.HealthCheck()
info, err := client.GetServerInfo()
examples, err := client.GetExamples()

// MCP Tools
tools, err := client.ListMCPTools()
result, err := client.CallMCPTool("tool_name", arguments)

// Financial Analysis
result, err := client.AnalyzePrice("AAPL", "2024-01-01", "2024-12-31", true)
result, err := client.AnalyzeSentiment("AAPL", "2024-01-01", "2024-12-31", []string{})
result, err := client.AnalyzeFundamentals("AAPL", true, true, true, 4)
result, err := client.AnalyzeMacro("AAPL", true, true)
result, err := client.GenerateCompanyProfile("AAPL")
result, err := client.AnalyzeRegulatoryCompliance("AAPL", true, true)

// Smart Query
result, err := client.QueryAnalyze("Analyze AAPL comprehensively", "parallel")
result, err := client.APIAnalyze("Analyze AAPL comprehensively", "parallel")

// Multi-Tool Execution
tools := []string{"analyze_price", "analyze_fundamentals"}
params := map[string]interface{}{"symbol": "AAPL"}
result, err := client.ExecuteTools(tools, params, "parallel")
```

#### Usage Patterns

```go
// Standard usage
client := NewClient("http://localhost:8080", "", 120)
result, err := client.AnalyzePrice("AAPL", "2024-01-01", "2024-12-31", true)
if err != nil {
    log.Printf("Analysis failed: %v", err)
    return
}
fmt.Printf("Result: %+v\n", result)

// Custom configuration
client := NewClient("http://custom-server:8080", "my_session", 300)
```

---

## CLI Implementation Details

### Python CLI (`cli.py`)

The `FinancialAnalysisCLI` class wraps the client library for command-line usage:

#### Architecture

- **Command Dispatcher**: Dynamic method resolution based on command names
- **Error Handling**: Graceful error messages for invalid commands/arguments
- **Client Integration**: Uses `FinancialAnalysisClient` internally

#### Command Categories

```python
# Basic health & info
def health(self)                    # Simple health check
def server_info(self)               # Server information
def health_check(self)              # Detailed health example
def list_tools(self)                # Available tools example
def date_utilities(self)            # Date formatting utilities

# Analysis commands
def analyze_price(self, symbol, start_date, end_date)
def analyze_sentiment(self, symbol, start_date, end_date)
def analyze_fundamentals(self, symbol, quarters="4")
def analyze_macro(self, symbol)
def company_profile(self, symbol)
def regulatory_compliance(self, symbol)

# Composite analyses
def comprehensive(self, symbol, start_date, end_date)
def due_diligence(self, symbol)
def sentiment_analysis(self, symbol, start_date, end_date)

# Tool management
def tools(self)                     # List MCP tools
def discover_tools(self)            # Discover available tools
def manual(self, symbol)            # Manual multi-tool execution
```

### Go CLI (`cli.go`)

The Go CLI provides a lightweight, direct HTTP interface:

#### Architecture

- **Direct HTTP Calls**: No intermediate client library
- **Simple Functions**: One function per command
- **JSON Payloads**: Manual payload construction for MCP calls

#### Implementation Pattern

```go
// Helper functions
func get(url string)                    // GET request helper
func post(url string, payload interface{}) // POST request helper

// Command implementations
func health()                           // GET /health
func serverInfo()                       // GET /info
func analyzePrice(symbol, start, end string) // POST /mcp/call
func tools()                            // GET /mcp/tools
func comprehensive(symbol, start, end string) // POST /mcp/call
```

---

## Key Points & Best Practices

### Data Formats

- **Date Format**: Always `YYYY-MM-DD` (ISO 8601)
- **Symbols**: Uppercase stock symbols (AAPL, MSFT, TSLA)
- **Timeouts**: Default 120 seconds, configurable
- **Session IDs**: Auto-generated or custom

### Execution Modes

- **`parallel`**: Runs tools simultaneously (default, faster)
- **`sequential`**: Runs tools in order (slower, but ordered)

### Error Handling

- **Python**: `FinancialAnalysisClientError` exceptions
- **Go**: Standard error return values
- **HTTP Errors**: Automatic status code checking
- **JSON Errors**: Graceful parsing error handling

### Performance Considerations

- **Connection Reuse**: Both clients pool HTTP connections
- **Timeout Management**: Configurable timeouts for long-running analyses
- **Session Tracking**: Enables server-side optimization
- **Parallel Execution**: Default mode for faster multi-tool operations

### Extensibility

- **Python CLI**: Add methods to `FinancialAnalysisCLI` class
- **Go CLI**: Add case statements in main switch
- **Client Libraries**: Extend with new endpoint methods
- **Custom Tools**: Register new MCP tools on server side

---

## Recommended Usage Patterns

### For Quick Analysis

```bash
# Python
python cli.py health && python cli.py analyze_price AAPL 2024-01-01 2024-12-31

# Go
./fincli health && ./fincli analyze_price AAPL 2024-01-01 2024-12-31
```

### For Comprehensive Analysis

```bash
# Most comprehensive results
./fincli comprehensive AAPL 2024-01-01 2024-12-31
python cli.py comprehensive AAPL 2024-01-01 2024-12-31
```

### For Programmatic Integration

```python
# Python library usage
with FinancialAnalysisClient() as client:
    if client.is_healthy():
        result = client.comprehensive_analysis("AAPL", "2024-01-01", "2024-12-31")
        # Process result...
```

```go
// Go library usage
client := NewClient("http://localhost:8080", "", 120)
result, err := client.QueryAnalyze("Comprehensive analysis of AAPL", "parallel")
if err != nil {
    log.Fatal(err)
}
// Process result...
```
