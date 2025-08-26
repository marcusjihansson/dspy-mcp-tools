// server.go
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
)

// -------------------- Types --------------------

type MCPTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

type MCPListToolsResponse struct {
	Tools []MCPTool `json:"tools"`
}

type MCPCallToolRequest struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

type MCPCallToolResponse struct {
	Content []map[string]interface{} `json:"content"`
	IsError bool                     `json:"isError"`
}

type QueryRequest struct {
	Query         string `json:"query"`
	ExecutionMode string `json:"execution_mode,omitempty"`
	SessionID     string `json:"session_id,omitempty"`
}

type QueryAnalysisResponse struct {
	SuggestedTools []string               `json:"suggested_tools"`
	ExecutionMode  string                 `json:"execution_mode"`
	Reasoning      string                 `json:"reasoning"`
	Parameters     map[string]interface{} `json:"parameters"`
}

type ToolExecutionRequest struct {
	Tools         []string               `json:"tools"`
	Parameters    map[string]interface{} `json:"parameters"`
	ExecutionMode string                 `json:"execution_mode"`
}

type ExecutionResponse struct {
	ExecutionMode string                 `json:"execution_mode"`
	Results       map[string]interface{} `json:"results"`
	Timestamp     string                 `json:"timestamp"`
}

type ToolExecutionResult struct {
	ToolName      string                 `json:"tool_name"`
	Success       bool                   `json:"success"`
	Result        map[string]interface{} `json:"result,omitempty"`
	Error         string                 `json:"error,omitempty"`
	ExecutionTime time.Duration          `json:"execution_time"`
}

type QueryAnalysisReport struct {
	Query           string                `json:"query"`
	SessionID       string                `json:"session_id"`
	ExecutionMode   string                `json:"execution_mode"`
	ToolsExecuted   []ToolExecutionResult `json:"tools_executed"`
	Summary         string                `json:"summary"`
	KeyFindings     []string              `json:"key_findings"`
	Recommendations []string              `json:"recommendations"`
	DataSources     []string              `json:"data_sources"`
	ExecutionTime   time.Duration         `json:"total_execution_time"`
	Timestamp       string                `json:"timestamp"`
}

type Config struct {
	Port       string        `json:"port"`
	PythonHost string        `json:"python_host"`
	PythonPort string        `json:"python_port"`
	Timeout    time.Duration `json:"timeout"`
}

// -------------------- Session management --------------------

type Session struct {
	ID        string
	CreatedAt time.Time
	LastUsed  time.Time
	Client    *FinancialAnalysisClient
	mutex     sync.RWMutex
}

type SessionManager struct {
	sessions map[string]*Session
	mutex    sync.RWMutex
	config   *Config
}

func NewSessionManager(config *Config) *SessionManager {
	sm := &SessionManager{
		sessions: make(map[string]*Session),
		config:   config,
	}
	go sm.cleanupSessions()
	return sm
}

func (sm *SessionManager) GetOrCreateSession(sessionID string) (*Session, error) {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	if sessionID == "" {
		sessionID = fmt.Sprintf("session_%d", time.Now().UnixNano())
	}

	session, exists := sm.sessions[sessionID]
	if !exists {
		client := NewFinancialAnalysisClient(
			fmt.Sprintf("http://%s:%s", sm.config.PythonHost, sm.config.PythonPort),
		)
		session = &Session{
			ID:        sessionID,
			CreatedAt: time.Now(),
			LastUsed:  time.Now(),
			Client:    client,
		}
		sm.sessions[sessionID] = session
		log.Printf("Created new session: %s", sessionID)
	} else {
		session.LastUsed = time.Now()
	}
	return session, nil
}

func (sm *SessionManager) Count() int {
	sm.mutex.RLock()
	defer sm.mutex.RUnlock()
	return len(sm.sessions)
}

func (sm *SessionManager) cleanupSessions() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		sm.mutex.Lock()
		cutoff := time.Now().Add(-30 * time.Minute)
		for id, session := range sm.sessions {
			if session.LastUsed.Before(cutoff) {
				delete(sm.sessions, id)
				log.Printf("Cleaned up expired session: %s", id)
			}
		}
		sm.mutex.Unlock()
	}
}

// -------------------- HTTP response recorder --------------------

type responseRecorder struct {
	http.ResponseWriter
	statusCode int
}

func (rec *responseRecorder) WriteHeader(statusCode int) {
	rec.statusCode = statusCode
	rec.ResponseWriter.WriteHeader(statusCode)
}

// -------------------- Server --------------------

type Server struct {
	config         *Config
	httpClient     *http.Client
	sessionManager *SessionManager
}

func NewServer(config *Config) *Server {
	httpClient := &http.Client{
		Timeout: config.Timeout,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 20,
			IdleConnTimeout:     90 * time.Second,
			TLSHandshakeTimeout: 10 * time.Second,
		},
	}
	return &Server{
		config:         config,
		httpClient:     httpClient,
		sessionManager: NewSessionManager(config),
	}
}

func (s *Server) respondWithJSON(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(data); err != nil {
		log.Printf("Failed to encode JSON response: %v", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
	}
}

func (s *Server) respondWithError(w http.ResponseWriter, message string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"success": false,
		"error":   message,
	})
}

// -------------------- FinancialAnalysisClient --------------------

type FinancialAnalysisClient struct {
	FastAPIURL    string
	HTTPClient    *http.Client
	toolsCache    map[string]MCPTool
	toolsCacheExp time.Time
	mutex         sync.RWMutex
}

func NewFinancialAnalysisClient(fastAPIURL string) *FinancialAnalysisClient {
	return &FinancialAnalysisClient{
		FastAPIURL: fastAPIURL,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        10,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		toolsCache: make(map[string]MCPTool),
	}
}

func (c *FinancialAnalysisClient) ListMCPTools() (*MCPListToolsResponse, error) {
	c.mutex.RLock()
	if time.Now().Before(c.toolsCacheExp) && len(c.toolsCache) > 0 {
		tools := make([]MCPTool, 0, len(c.toolsCache))
		for _, tool := range c.toolsCache {
			tools = append(tools, tool)
		}
		c.mutex.RUnlock()
		return &MCPListToolsResponse{Tools: tools}, nil
	}
	c.mutex.RUnlock()

	resp, err := c.HTTPClient.Get(c.FastAPIURL + "/mcp/tools")
	if err != nil {
		return nil, fmt.Errorf("failed to fetch MCP tools: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("MCP tools request failed with status: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var toolsResponse MCPListToolsResponse
	if err := json.Unmarshal(body, &toolsResponse); err != nil {
		return nil, err
	}

	c.mutex.Lock()
	c.toolsCache = make(map[string]MCPTool)
	for _, tool := range toolsResponse.Tools {
		c.toolsCache[tool.Name] = tool
	}
	c.toolsCacheExp = time.Now().Add(5 * time.Minute)
	c.mutex.Unlock()

	return &toolsResponse, nil
}

func (c *FinancialAnalysisClient) CallMCPTool(toolName string, arguments map[string]interface{}) (*MCPCallToolResponse, error) {
	reqBody := MCPCallToolRequest{
		Name:      toolName,
		Arguments: arguments,
	}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	resp, err := c.HTTPClient.Post(c.FastAPIURL+"/mcp/call", "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var callResponse MCPCallToolResponse
	if err := json.Unmarshal(body, &callResponse); err != nil {
		return nil, err
	}
	return &callResponse, nil
}

// executeTools calls the Python /tools/execute (legacy bulk execution)
func (c *FinancialAnalysisClient) executeTools(tools []string, parameters map[string]interface{}, executionMode string) (*ExecutionResponse, error) {
	reqBody := ToolExecutionRequest{
		Tools:         tools,
		Parameters:    parameters,
		ExecutionMode: executionMode,
	}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	resp, err := c.HTTPClient.Post(c.FastAPIURL+"/tools/execute", "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("execution request failed with status: %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var execution ExecutionResponse
	if err := json.Unmarshal(body, &execution); err != nil {
		return nil, err
	}
	return &execution, nil
}

// analyzeQuery for delegating to Python /query/analyze
func (c *FinancialAnalysisClient) analyzeQuery(query string, executionMode string) (*QueryAnalysisResponse, error) {
	reqBody := QueryRequest{
		Query:         query,
		ExecutionMode: executionMode,
	}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	resp, err := c.HTTPClient.Post(c.FastAPIURL+"/query/analyze", "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("analysis request failed with status: %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var analysis QueryAnalysisResponse
	if err := json.Unmarshal(body, &analysis); err != nil {
		return nil, err
	}
	return &analysis, nil
}

// AnalyzeAndExecute (legacy) - returns ExecutionResponse
func (c *FinancialAnalysisClient) AnalyzeAndExecute(query string, executionMode string) (*ExecutionResponse, error) {
	analysis, err := c.analyzeQuery(query, executionMode)
	if err != nil {
		return nil, fmt.Errorf("query analysis failed: %w", err)
	}
	log.Printf("Query Analysis - Tools: %v, Mode: %s, Reasoning: %s",
		analysis.SuggestedTools, analysis.ExecutionMode, analysis.Reasoning)
	execution, err := c.executeTools(analysis.SuggestedTools, analysis.Parameters, analysis.ExecutionMode)
	if err != nil {
		return nil, fmt.Errorf("tool execution failed: %w", err)
	}
	return execution, nil
}

// Enhanced Analyze + detailed report
func (c *FinancialAnalysisClient) AnalyzeAndExecuteWithReport(query string, executionMode string, sessionID string) (*QueryAnalysisReport, error) {
	startTime := time.Now()
	analysis, err := c.analyzeQuery(query, executionMode)
	if err != nil {
		return nil, fmt.Errorf("query analysis failed: %w", err)
	}
	log.Printf("Session %s - Query Analysis - Tools: %v, Mode: %s",
		sessionID, analysis.SuggestedTools, analysis.ExecutionMode)
	toolResults, err := c.executeToolsWithDetails(analysis.SuggestedTools, analysis.Parameters, analysis.ExecutionMode)
	if err != nil {
		return nil, fmt.Errorf("tool execution failed: %w", err)
	}
	report := c.generateAnalysisReport(query, sessionID, analysis.ExecutionMode, toolResults, time.Since(startTime))
	return report, nil
}

// Helper: parse content item (attempt to unmarshal JSON text if present)
func parseMCPContentItem(content map[string]interface{}) map[string]interface{} {
	// If the content has "text" that contains JSON, parse it
	if txtVal, ok := content["text"]; ok {
		if txt, ok := txtVal.(string); ok && len(strings.TrimSpace(txt)) > 0 {
			trimmed := strings.TrimSpace(txt)
			// try to auto-detect JSON
			if strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[") {
				var parsed interface{}
				if err := json.Unmarshal([]byte(trimmed), &parsed); err == nil {
					switch p := parsed.(type) {
					case map[string]interface{}:
						return p
					default:
						return map[string]interface{}{"data": p}
					}
				}
			}
			// not JSON or failed to unmarshal -> return as text field
			return map[string]interface{}{"text": txt}
		}
	}
	// no text field or not parseable - just return content as-is
	return content
}

func (c *FinancialAnalysisClient) executeToolsWithDetails(tools []string, parameters map[string]interface{}, executionMode string) ([]ToolExecutionResult, error) {
	var results []ToolExecutionResult

	if len(tools) == 0 {
		return results, nil
	}

	if executionMode == "sequential" {
		for _, toolName := range tools {
			startTime := time.Now()
			result, err := c.CallMCPTool(toolName, parameters)

			tr := ToolExecutionResult{
				ToolName:      toolName,
				ExecutionTime: time.Since(startTime),
				Success:       err == nil && !(result != nil && result.IsError),
			}

			if err != nil {
				tr.Error = err.Error()
			} else if result == nil {
				tr.Success = false
				tr.Error = "empty response"
			} else if result.IsError {
				tr.Success = false
				// try to extract textual error content
				if len(result.Content) > 0 {
					if t0, ok := result.Content[0]["text"].(string); ok {
						tr.Error = t0
					} else {
						tr.Error = "tool returned error"
					}
				} else {
					tr.Error = "tool returned error"
				}
			} else {
				// success - parse content
				if len(result.Content) == 1 {
					tr.Result = parseMCPContentItem(result.Content[0])
				} else if len(result.Content) > 1 {
					agg := make(map[string]interface{})
					for i, cont := range result.Content {
						key := fmt.Sprintf("result_%d", i)
						agg[key] = parseMCPContentItem(cont)
					}
					tr.Result = agg
				}
			}

			results = append(results, tr)

			if !tr.Success {
				log.Printf("Tool %s failed in sequential mode: %s", toolName, tr.Error)
			}
		}
	} else {
		type te struct {
			idx    int
			result ToolExecutionResult
		}
		ch := make(chan te, len(tools))

		for i, toolName := range tools {
			go func(idx int, t string) {
				startTime := time.Now()
				result, err := c.CallMCPTool(t, parameters)
				tr := ToolExecutionResult{
					ToolName:      t,
					ExecutionTime: time.Since(startTime),
					Success:       err == nil && !(result != nil && result.IsError),
				}
				if err != nil {
					tr.Error = err.Error()
				} else if result == nil {
					tr.Success = false
					tr.Error = "empty response"
				} else if result.IsError {
					tr.Success = false
					if len(result.Content) > 0 {
						if t0, ok := result.Content[0]["text"].(string); ok {
							tr.Error = t0
						} else {
							tr.Error = "tool returned error"
						}
					} else {
						tr.Error = "tool returned error"
					}
				} else {
					if len(result.Content) == 1 {
						tr.Result = parseMCPContentItem(result.Content[0])
					} else if len(result.Content) > 1 {
						agg := make(map[string]interface{})
						for i, cont := range result.Content {
							key := fmt.Sprintf("result_%d", i)
							agg[key] = parseMCPContentItem(cont)
						}
						tr.Result = agg
					}
				}
				ch <- te{idx: idx, result: tr}
			}(i, toolName)
		}

		temp := make([]ToolExecutionResult, len(tools))
		for i := 0; i < len(tools); i++ {
			t := <-ch
			temp[t.idx] = t.result
		}
		results = append(results, temp...)
	}

	return results, nil
}

func extractKeyFindingsFromResult(m map[string]interface{}) []string {
	var findings []string
	// try top-level text fields
	if text, ok := m["text"].(string); ok && len(strings.TrimSpace(text)) > 0 {
		parts := strings.Split(text, ".")
		if len(parts) > 0 && len(strings.TrimSpace(parts[0])) > 5 {
			findings = append(findings, strings.TrimSpace(parts[0]))
		}
	}
	// check common nested patterns
	if data, ok := m["data"].(map[string]interface{}); ok {
		findings = append(findings, extractKeyFindingsFromResult(data)...)
	}
	// check "analysis"
	if a, ok := m["analysis"].(string); ok && len(a) > 0 {
		parts := strings.Split(a, ".")
		if len(parts) > 0 && len(strings.TrimSpace(parts[0])) > 5 {
			findings = append(findings, strings.TrimSpace(parts[0]))
		}
	}
	// check for numeric/label keys
	if price, ok := m["price"]; ok {
		findings = append(findings, fmt.Sprintf("Price data: %v", price))
	}
	if s, ok := m["sentiment_label"]; ok {
		findings = append(findings, fmt.Sprintf("Sentiment: %v", s))
	} else if ss, ok := m["sentiment_score"]; ok {
		findings = append(findings, fmt.Sprintf("Sentiment score: %v", ss))
	}
	return findings
}

func (c *FinancialAnalysisClient) generateAnalysisReport(query string, sessionID string, executionMode string, toolResults []ToolExecutionResult, totalTime time.Duration) *QueryAnalysisReport {
	report := &QueryAnalysisReport{
		Query:         query,
		SessionID:     sessionID,
		ExecutionMode: executionMode,
		ToolsExecuted: toolResults,
		ExecutionTime: totalTime,
		Timestamp:     time.Now().UTC().Format(time.RFC3339),
	}

	var summaryParts []string
	var keyFindings []string
	var recommendations []string
	var dataSources []string

	successfulTools := 0
	for _, result := range toolResults {
		if result.Success {
			successfulTools++
			dataSources = append(dataSources, result.ToolName)
			if result.Result != nil {
				keyFindings = append(keyFindings, extractKeyFindingsFromResult(result.Result)...)
			}
		} else {
			// Could capture errors in summary if needed
		}
	}

	if successfulTools > 0 {
		summaryParts = append(summaryParts, fmt.Sprintf("Successfully executed %d out of %d tools", successfulTools, len(toolResults)))
		summaryParts = append(summaryParts, fmt.Sprintf("Analysis completed in %v using %s execution mode", totalTime.Round(time.Millisecond), executionMode))
		if len(keyFindings) > 0 {
			summaryParts = append(summaryParts, "Key insights were extracted from financial data sources")
		}
	} else {
		summaryParts = append(summaryParts, "Analysis encountered issues - no tools executed successfully")
	}

	report.Summary = strings.Join(summaryParts, ". ")
	report.KeyFindings = keyFindings
	report.DataSources = dataSources

	if successfulTools > 0 {
		recommendations = append(recommendations, "Review the detailed tool results for comprehensive insights")
		if executionMode == "parallel" && len(toolResults) > 1 {
			recommendations = append(recommendations, "Consider sequential analysis for dependent data relationships")
		}
		if len(keyFindings) == 0 {
			recommendations = append(recommendations, "Consider refining query parameters for more specific insights")
		}
	} else {
		recommendations = append(recommendations, "Check tool availability and parameters")
		recommendations = append(recommendations, "Consider simplifying the query or using different tools")
	}
	report.Recommendations = recommendations
	return report
}

// -------------------- Middleware --------------------

func (s *Server) withMiddleware(handler http.HandlerFunc) http.HandlerFunc {
	return s.loggingMiddleware(s.corsMiddleware(s.errorRecoveryMiddleware(handler)))
}

func (s *Server) corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Session-ID")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		next(w, r)
	}
}

func (s *Server) loggingMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &responseRecorder{ResponseWriter: w, statusCode: http.StatusOK}
		next(rec, r)
		sessionID := r.Header.Get("X-Session-ID")
		if sessionID == "" {
			sessionID = "no-session"
		}
		log.Printf("[%s] %s %s %d %v", sessionID, r.Method, r.URL.Path, rec.statusCode, time.Since(start))
	}
}

func (s *Server) errorRecoveryMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				log.Printf("Panic recovered: %v", err)
				s.respondWithError(w, "Internal server error", http.StatusInternalServerError)
			}
		}()
		next(w, r)
	}
}

// -------------------- Handlers --------------------

func (s *Server) handleMCPListTools(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.respondWithError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	sessionID := r.Header.Get("X-Session-ID")
	session, err := s.sessionManager.GetOrCreateSession(sessionID)
	if err != nil {
		s.respondWithError(w, fmt.Sprintf("Failed to get session: %v", err), http.StatusInternalServerError)
		return
	}
	tools, err := session.Client.ListMCPTools()
	if err != nil {
		s.respondWithError(w, fmt.Sprintf("Failed to list tools: %v", err), http.StatusServiceUnavailable)
		return
	}
	s.respondWithJSON(w, tools)
}

func (s *Server) handleMCPCallTool(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.respondWithError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req MCPCallToolRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondWithError(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Name == "" {
		s.respondWithError(w, "Tool name is required", http.StatusBadRequest)
		return
	}
	sessionID := r.Header.Get("X-Session-ID")
	session, err := s.sessionManager.GetOrCreateSession(sessionID)
	if err != nil {
		s.respondWithError(w, fmt.Sprintf("Failed to get session: %v", err), http.StatusInternalServerError)
		return
	}
	result, err := session.Client.CallMCPTool(req.Name, req.Arguments)
	if err != nil {
		s.respondWithError(w, "Failed to call MCP tool: "+err.Error(), http.StatusInternalServerError)
		return
	}
	s.respondWithJSON(w, result)
}

func (s *Server) handleQueryAnalyze(w http.ResponseWriter, r *http.Request) {
	// Detailed report endpoint
	if r.Method != http.MethodPost {
		s.respondWithError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondWithError(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	// header X-Session-ID takes precedence over body.session_id
	sessionID := r.Header.Get("X-Session-ID")
	if sessionID == "" {
		sessionID = req.SessionID
	}
	session, err := s.sessionManager.GetOrCreateSession(sessionID)
	if err != nil {
		s.respondWithError(w, fmt.Sprintf("Failed to create session: %v", err), http.StatusInternalServerError)
		return
	}
	report, err := session.Client.AnalyzeAndExecuteWithReport(req.Query, req.ExecutionMode, session.ID)
	if err != nil {
		s.respondWithError(w, fmt.Sprintf("Analysis failed: %v", err), http.StatusInternalServerError)
		return
	}
	s.respondWithJSON(w, report)
}

func (s *Server) handleAPIAnalyze(w http.ResponseWriter, r *http.Request) {
	// Legacy analyze endpoint returning ExecutionResponse
	if r.Method != http.MethodPost {
		s.respondWithError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondWithError(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	sessionID := r.Header.Get("X-Session-ID")
	if sessionID == "" {
		sessionID = req.SessionID
	}
	session, err := s.sessionManager.GetOrCreateSession(sessionID)
	if err != nil {
		s.respondWithError(w, fmt.Sprintf("Failed to get session: %v", err), http.StatusInternalServerError)
		return
	}
	result, err := session.Client.AnalyzeAndExecute(req.Query, req.ExecutionMode)
	if err != nil {
		s.respondWithError(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.respondWithJSON(w, result)
}

func (s *Server) handleToolsExecute(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.respondWithError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req ToolExecutionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondWithError(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	sessionID := r.Header.Get("X-Session-ID")
	session, err := s.sessionManager.GetOrCreateSession(sessionID)
	if err != nil {
		s.respondWithError(w, fmt.Sprintf("Failed to get session: %v", err), http.StatusInternalServerError)
		return
	}
	result, err := session.Client.executeTools(req.Tools, req.Parameters, req.ExecutionMode)
	if err != nil {
		s.respondWithError(w, fmt.Sprintf("Tool execution failed: %v", err), http.StatusInternalServerError)
		return
	}
	s.respondWithJSON(w, result)
}

func (s *Server) handleToolsDiscovery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.respondWithError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	pythonURL := fmt.Sprintf("http://%s:%s/tools/discover", s.config.PythonHost, s.config.PythonPort)
	resp, err := s.httpClient.Get(pythonURL)
	if err != nil {
		s.respondWithError(w, "Failed to connect to Python service: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		s.respondWithError(w, "Failed to read response: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(body)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.respondWithError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	pythonURL := fmt.Sprintf("http://%s:%s/health", s.config.PythonHost, s.config.PythonPort)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", pythonURL, nil)
	resp, err := s.httpClient.Do(req)
	pythonHealthy := err == nil && resp != nil && resp.StatusCode == http.StatusOK
	if resp != nil {
		resp.Body.Close()
	}
	status := map[string]interface{}{
		"status":            "healthy",
		"timestamp":         time.Now().UTC(),
		"python_service":    pythonHealthy,
		"mcp_support":       true,
		"go_server_version": "2.0.0",
		"sessions":          s.sessionManager.Count(),
	}
	if !pythonHealthy {
		status["status"] = "degraded"
		w.WriteHeader(http.StatusServiceUnavailable)
	}
	s.respondWithJSON(w, status)
}

func (s *Server) handleServerInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.respondWithError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	info := map[string]interface{}{
		"name":        "Go MCP Financial Analysis Server",
		"version":     "2.0.0",
		"description": "Golang server providing MCP-compliant financial analysis tools with auto-discovery",
		"capabilities": []string{
			"mcp_protocol",
			"auto_discovery",
			"parallel_execution",
			"sequential_execution",
			"query_analysis",
			"tool_chaining",
		},
	}
	s.respondWithJSON(w, info)
}

func (s *Server) handleExamples(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.respondWithError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	examples := map[string]interface{}{
		"mcp_examples": map[string]interface{}{
			"list_all_tools": map[string]interface{}{
				"method":      "GET",
				"endpoint":    "/mcp/tools",
				"description": "Get all available MCP tools with schemas",
			},
			"call_single_tool": map[string]interface{}{
				"method":   "POST",
				"endpoint": "/mcp/call",
				"body": map[string]interface{}{
					"name": "analyze_price",
					"arguments": map[string]interface{}{
						"symbol": "AAPL",
					},
				},
				"description": "Call a specific MCP tool directly",
			},
		},
		"smart_analysis_examples": map[string]interface{}{
			"comprehensive_analysis": map[string]interface{}{
				"method":   "POST",
				"endpoint": "/api/analyze",
				"body": map[string]interface{}{
					"query":          "Give me a comprehensive analysis of AAPL",
					"execution_mode": "",
				},
				"description": "Auto-discover and execute multiple tools in parallel",
			},
		},
	}
	s.respondWithJSON(w, examples)
}

// -------------------- Config helpers & main --------------------

func getEnvOrDefault(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}

func loadConfig() *Config {
	cfg := &Config{
		Port:       getEnvOrDefault("PORT", "8080"),
		PythonHost: getEnvOrDefault("PYTHON_HOST", "localhost"),
		PythonPort: getEnvOrDefault("PYTHON_PORT", "8000"),
		Timeout:    30 * time.Second,
	}
	if timeoutStr := os.Getenv("TIMEOUT_SECONDS"); timeoutStr != "" {
		if tSec, err := time.ParseDuration(timeoutStr + "s"); err == nil {
			cfg.Timeout = tSec
		}
	}
	return cfg
}

func main() {
	config := loadConfig()
	server := NewServer(config)

	mux := http.NewServeMux()
	mux.HandleFunc("/mcp/tools", server.withMiddleware(server.handleMCPListTools))
	mux.HandleFunc("/mcp/call", server.withMiddleware(server.handleMCPCallTool))
	mux.HandleFunc("/query/analyze", server.withMiddleware(server.handleQueryAnalyze))
	mux.HandleFunc("/api/analyze", server.withMiddleware(server.handleAPIAnalyze))
	mux.HandleFunc("/api/tools/discover", server.withMiddleware(server.handleToolsDiscovery))
	mux.HandleFunc("/tools/execute", server.withMiddleware(server.handleToolsExecute))
	mux.HandleFunc("/health", server.withMiddleware(server.handleHealth))
	mux.HandleFunc("/info", server.withMiddleware(server.handleServerInfo))
	mux.HandleFunc("/examples", server.withMiddleware(server.handleExamples))

	httpServer := &http.Server{
		Addr:         ":" + config.Port,
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		log.Printf("🚀 Server starting on port %s", config.Port)
		log.Printf("🐍 Python FastAPI expected at %s:%s", config.PythonHost, config.PythonPort)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed to start: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("🛑 Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}
	log.Println("✅ Server exited gracefully")
}
