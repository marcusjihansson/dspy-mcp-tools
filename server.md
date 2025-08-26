# Go Financial Analysis Server API Documentation

This document provides comprehensive information about the Go server endpoints and their usage.

## Architecture Overview

The Go server acts as a proxy and enhancement layer over the Python FastAPI server, providing:

- Session management
- Enhanced error handling
- Detailed reporting
- MCP protocol compliance

## Server Setup

```bash
# Terminal 1 - Python FastAPI (port 8000)
python fast_server.py

# Terminal 2 - Go Server (port 8080)
go run server.go
```

## Available Endpoints

### MCP Protocol Endpoints

- `GET /mcp/tools` - List all available MCP tools
- `POST /mcp/call` - Execute a specific MCP tool

### Enhanced Analysis Endpoints

- `POST /query/analyze` - Smart query analysis with detailed reports
- `POST /api/analyze` - Legacy analyze endpoint
- `POST /tools/execute` - Execute multiple tools

### Utility Endpoints

- `GET /health` - Health check
- `GET /info` - Server information
- `GET /examples` - API usage examples

## Detailed Tool Schemas

### 1. `analyze_price`

**Required:** `symbol`  
**Optional:** `start_date`, `end_date`, `include_technical`

```bash
curl -X POST http://localhost:8080/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "analyze_price",
    "arguments": {
      "symbol": "AAPL",
      "start_date": "2024-01-01",
      "end_date": "2024-12-31",
      "include_technical": true
    }
  }'
```

### 2. `analyze_sentiment`

**Required:** `symbol`  
**Optional:** `start_date`, `end_date`, `sources`

```bash
curl -X POST http://localhost:8080/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "analyze_sentiment",
    "arguments": {
      "symbol": "AAPL",
      "start_date": "2024-01-01",
      "end_date": "2024-12-31",
      "sources": ["reuters", "bloomberg", "wsj"]
    }
  }'
```

### 3. `analyze_fundamentals`

**Required:** `symbol`  
**Optional:** `include_balance_sheet`, `include_cash_flow`, `include_earnings`, `quarters`

```bash
curl -X POST http://localhost:8080/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "analyze_fundamentals",
    "arguments": {
      "symbol": "AAPL",
      "include_balance_sheet": true,
      "include_cash_flow": true,
      "include_earnings": true,
      "quarters": 4
    }
  }'
```

### 4. `analyze_macro`

**Required:** `symbol`  
**Optional:** `include_sector_analysis`, `include_economic_indicators`

```bash
curl -X POST http://localhost:8080/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "analyze_macro",
    "arguments": {
      "symbol": "AAPL",
      "include_sector_analysis": true,
      "include_economic_indicators": true
    }
  }'
```

### 5. `generate_company_profile`

**Required:** `symbol`

```bash
curl -X POST http://localhost:8080/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "generate_company_profile",
    "arguments": {
      "symbol": "AAPL"
    }
  }'
```

### 6. `analyze_regulatory_compliance`

**Required:** `symbol`  
**Optional:** `include_ai_regulations`, `include_sector_regulations`

```bash
curl -X POST http://localhost:8080/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "analyze_regulatory_compliance",
    "arguments": {
      "symbol": "AAPL",
      "include_ai_regulations": true,
      "include_sector_regulations": true
    }
  }'
```

## Smart Query Analysis Examples

### Comprehensive Analysis with Dates

```bash
curl -X POST http://localhost:8080/query/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Give me a comprehensive analysis of AAPL from 2024-01-01 to 2024-12-31",
    "execution_mode": "parallel"
  }'
```

### Sequential Analysis for Due Diligence

```bash
curl -X POST http://localhost:8080/query/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Perform step by step due diligence analysis for MSFT from 2024-06-01 to 2024-12-31",
    "execution_mode": "sequential"
  }'
```

### Sentiment-focused Analysis

```bash
curl -X POST http://localhost:8080/query/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze sentiment and news for TSLA from 2024-10-01 to 2024-12-31"
  }'
```

## Manual Tool Execution with Multiple Tools

### Execute Multiple Tools in Parallel

```bash
curl -X POST http://localhost:8080/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tools": ["analyze_price", "analyze_sentiment", "analyze_fundamentals"],
    "parameters": {
      "symbol": "AAPL",
      "start_date": "2024-01-01",
      "end_date": "2024-12-31"
    },
    "execution_mode": "parallel"
  }'
```

### Execute Tools Sequentially

```bash
curl -X POST http://localhost:8080/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tools": ["generate_company_profile", "analyze_regulatory_compliance"],
    "parameters": {
      "symbol": "AAPL"
    },
    "execution_mode": "sequential"
  }'
```

## Session Management

You can also use session headers for tracking:

```bash
curl -X POST http://localhost:8080/query/analyze \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-analysis-session-123" \
  -d '{
    "query": "Comprehensive AAPL analysis from 2024-01-01 to 2024-12-31"
  }'
```

## Utility Endpoints

### Health Check

```bash
curl -X GET http://localhost:8080/health
```

### Server Information

```bash
curl -X GET http://localhost:8080/info
```

### Available Examples

```bash
curl -X GET http://localhost:8080/examples
```

### Tool Discovery

```bash
curl -X GET http://localhost:8080/api/tools/discover
```

### List All MCP Tools

```bash
curl -X GET http://localhost:8080/mcp/tools
```

## Key Points:

1. **Date Format:** Always use `YYYY-MM-DD` format for dates
2. **Symbol:** Stock symbols should be uppercase (AAPL, MSFT, TSLA, etc.)
3. **Smart Query:** The query analyzer can extract symbols and dates from natural language
4. **Execution Modes:**
   - `parallel` (default) - runs tools simultaneously
   - `sequential` - runs tools in dependency order
5. **Sessions:** Use `X-Session-ID` header for session tracking

## Server Ports

- **Python FastAPI Server:** Port 8000
- **Go Server:** Port 8080 (recommended for enhanced features)

## Recommended Usage

The `/query/analyze` endpoint is recommended as it provides the most comprehensive analysis with detailed reporting, including execution times, tool results, and recommendations. The `X-Session-ID` header is optional but useful for tracking sessions.
