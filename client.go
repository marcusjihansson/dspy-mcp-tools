package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// FinancialAnalysisClient is a Go client for the Financial Analysis Server
type FinancialAnalysisClient struct {
	BaseURL    string
	Timeout    time.Duration
	SessionID  string
	httpClient *http.Client
}

// NewClient creates a new FinancialAnalysisClient
func NewClient(baseURL, sessionID string, timeoutSeconds int) *FinancialAnalysisClient {
	if strings.HasSuffix(baseURL, "/") {
		baseURL = strings.TrimSuffix(baseURL, "/")
	}
	if sessionID == "" {
		sessionID = fmt.Sprintf("client_session_%d", time.Now().Unix())
	}
	return &FinancialAnalysisClient{
		BaseURL:    baseURL,
		Timeout:    time.Duration(timeoutSeconds) * time.Second,
		SessionID:  sessionID,
		httpClient: &http.Client{Timeout: time.Duration(timeoutSeconds) * time.Second},
	}
}

// generic request function
func (c *FinancialAnalysisClient) makeRequest(method, endpoint string, body interface{}) (map[string]interface{}, error) {
	url := c.BaseURL + endpoint

	var buf bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&buf).Encode(body); err != nil {
			return nil, fmt.Errorf("failed to encode request body: %w", err)
		}
	}

	req, err := http.NewRequest(method, url, &buf)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Session-ID", c.SessionID)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("server returned status %d", resp.StatusCode)
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode JSON: %w", err)
	}

	return result, nil
}

// ----------------- Health & Info -----------------
func (c *FinancialAnalysisClient) HealthCheck() (map[string]interface{}, error) {
	return c.makeRequest("GET", "/health", nil)
}

func (c *FinancialAnalysisClient) GetServerInfo() (map[string]interface{}, error) {
	return c.makeRequest("GET", "/info", nil)
}

func (c *FinancialAnalysisClient) GetExamples() (map[string]interface{}, error) {
	return c.makeRequest("GET", "/examples", nil)
}

// ----------------- MCP Tools -----------------
func (c *FinancialAnalysisClient) ListMCPTools() (map[string]interface{}, error) {
	return c.makeRequest("GET", "/mcp/tools", nil)
}

func (c *FinancialAnalysisClient) CallMCPTool(toolName string, arguments map[string]interface{}) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"name":      toolName,
		"arguments": arguments,
	}
	return c.makeRequest("POST", "/mcp/call", payload)
}

// ----------------- Financial Analysis -----------------
func (c *FinancialAnalysisClient) AnalyzePrice(symbol, startDate, endDate string) (map[string]interface{}, error) {
	args := map[string]interface{}{
		"symbol": strings.ToUpper(symbol),
	}
	if startDate != "" {
		args["start_date"] = startDate
	}
	if endDate != "" {
		args["end_date"] = endDate
	}
	return c.CallMCPTool("analyze_price", args)
}

func (c *FinancialAnalysisClient) AnalyzeSentiment(symbol, startDate, endDate string, sources []string) (map[string]interface{}, error) {
	args := map[string]interface{}{
		"symbol": strings.ToUpper(symbol),
	}
	if startDate != "" {
		args["start_date"] = startDate
	}
	if endDate != "" {
		args["end_date"] = endDate
	}
	if len(sources) > 0 {
		args["sources"] = sources
	}
	return c.CallMCPTool("analyze_sentiment", args)
}

func (c *FinancialAnalysisClient) AnalyzeFundamentals(symbol string, includeBalanceSheet, includeCashFlow, includeEarnings bool, quarters int) (map[string]interface{}, error) {
	args := map[string]interface{}{
		"symbol":                strings.ToUpper(symbol),
		"include_balance_sheet": includeBalanceSheet,
		"include_cash_flow":     includeCashFlow,
		"include_earnings":      includeEarnings,
		"quarters":              quarters,
	}
	return c.CallMCPTool("analyze_fundamentals", args)
}

func (c *FinancialAnalysisClient) AnalyzeMacro(symbol string, includeSectorAnalysis, includeEconomicIndicators bool) (map[string]interface{}, error) {
	args := map[string]interface{}{
		"symbol":                      strings.ToUpper(symbol),
		"include_sector_analysis":     includeSectorAnalysis,
		"include_economic_indicators": includeEconomicIndicators,
	}
	return c.CallMCPTool("analyze_macro", args)
}

func (c *FinancialAnalysisClient) GenerateCompanyProfile(symbol string) (map[string]interface{}, error) {
	args := map[string]interface{}{
		"symbol": strings.ToUpper(symbol),
	}
	return c.CallMCPTool("generate_company_profile", args)
}

func (c *FinancialAnalysisClient) AnalyzeRegulatoryCompliance(symbol string, includeAIRegulations, includeSectorRegulations bool) (map[string]interface{}, error) {
	args := map[string]interface{}{
		"symbol":                     strings.ToUpper(symbol),
		"include_ai_regulations":     includeAIRegulations,
		"include_sector_regulations": includeSectorRegulations,
	}
	return c.CallMCPTool("analyze_regulatory_compliance", args)
}

// ----------------- Smart Query -----------------
func (c *FinancialAnalysisClient) QueryAnalyze(query, executionMode string) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"query": query,
	}
	if executionMode != "" {
		payload["execution_mode"] = executionMode
	}
	return c.makeRequest("POST", "/query/analyze", payload)
}

func (c *FinancialAnalysisClient) APIAnalyze(query, executionMode string) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"query": query,
	}
	if executionMode != "" {
		payload["execution_mode"] = executionMode
	}
	return c.makeRequest("POST", "/api/analyze", payload)
}

// ----------------- Execute Tools -----------------
func (c *FinancialAnalysisClient) ExecuteTools(tools []string, parameters map[string]interface{}, executionMode string) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"tools":          tools,
		"parameters":     parameters,
		"execution_mode": executionMode,
	}
	return c.makeRequest("POST", "/tools/execute", payload)
}

func (c *FinancialAnalysisClient) DiscoverTools() (map[string]interface{}, error) {
	return c.makeRequest("GET", "/api/tools/discover", nil)
}
