package main

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// FinancialAnalysisCLI wraps the client with CLI functionality
type FinancialAnalysisCLI struct {
	client *FinancialAnalysisClient
}

// NewCLI creates a new CLI instance with default client settings
func NewCLI() *FinancialAnalysisCLI {
	// Default settings - can be made configurable via flags
	baseURL := "http://localhost:8080"
	sessionID := ""
	timeoutSeconds := 90

	client := NewClient(baseURL, sessionID, timeoutSeconds)
	return &FinancialAnalysisCLI{client: client}
}

// prettyPrint formats and prints JSON response
func (cli *FinancialAnalysisCLI) prettyPrint(result map[string]interface{}, err error) {
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	jsonBytes, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Printf("Error formatting output: %v\n", err)
		return
	}
	fmt.Println(string(jsonBytes))
}

// ---------------- BASIC HEALTH & INFO ---------------- //

func (cli *FinancialAnalysisCLI) Health() {
	result, err := cli.client.HealthCheck()
	if err != nil {
		fmt.Println("❌ Server is not healthy")
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Println("✅ Server is healthy")
	cli.prettyPrint(result, nil)
}

func (cli *FinancialAnalysisCLI) ServerInfo() {
	result, err := cli.client.GetServerInfo()
	cli.prettyPrint(result, err)
}

func (cli *FinancialAnalysisCLI) Examples() {
	result, err := cli.client.GetExamples()
	cli.prettyPrint(result, err)
}

// ---------------- ANALYSIS COMMANDS ---------------- //

func (cli *FinancialAnalysisCLI) AnalyzePrice(symbol, startDate, endDate string) {
	result, err := cli.client.AnalyzePrice(symbol, startDate, endDate)
	cli.prettyPrint(result, err)
}

func (cli *FinancialAnalysisCLI) AnalyzeSentiment(symbol, startDate, endDate string) {
	sources := []string{} // Default empty, could be made configurable
	result, err := cli.client.AnalyzeSentiment(symbol, startDate, endDate, sources)
	cli.prettyPrint(result, err)
}

func (cli *FinancialAnalysisCLI) AnalyzeFundamentals(symbol, quartersStr string) {
	quarters, err := strconv.Atoi(quartersStr)
	if err != nil {
		fmt.Printf("Error: quarters must be a number, got '%s'\n", quartersStr)
		return
	}

	result, err := cli.client.AnalyzeFundamentals(symbol, true, true, true, quarters)
	cli.prettyPrint(result, err)
}

func (cli *FinancialAnalysisCLI) AnalyzeMacro(symbol string) {
	result, err := cli.client.AnalyzeMacro(symbol, true, true)
	cli.prettyPrint(result, err)
}

func (cli *FinancialAnalysisCLI) CompanyProfile(symbol string) {
	result, err := cli.client.GenerateCompanyProfile(symbol)
	cli.prettyPrint(result, err)
}

func (cli *FinancialAnalysisCLI) RegulatoryCompliance(symbol string) {
	result, err := cli.client.AnalyzeRegulatoryCompliance(symbol, true, true)
	cli.prettyPrint(result, err)
}

// ---------------- QUERY / COMPOSITE COMMANDS ---------------- //

func (cli *FinancialAnalysisCLI) Comprehensive(symbol, startDate, endDate string) {
	query := fmt.Sprintf("comprehensive analysis of %s from %s to %s", symbol, startDate, endDate)
	result, err := cli.client.QueryAnalyze(query, "")
	cli.prettyPrint(result, err)
}

func (cli *FinancialAnalysisCLI) DueDiligence(symbol string) {
	query := fmt.Sprintf("due diligence analysis of %s", symbol)
	result, err := cli.client.QueryAnalyze(query, "")
	cli.prettyPrint(result, err)
}

func (cli *FinancialAnalysisCLI) SentimentAnalysis(symbol, startDate, endDate string) {
	query := fmt.Sprintf("sentiment analysis of %s from %s to %s", symbol, startDate, endDate)
	result, err := cli.client.QueryAnalyze(query, "")
	cli.prettyPrint(result, err)
}

// ---------------- TOOLS ---------------- //

func (cli *FinancialAnalysisCLI) Tools() {
	result, err := cli.client.ListMCPTools()
	cli.prettyPrint(result, err)
}

func (cli *FinancialAnalysisCLI) DiscoverTools() {
	result, err := cli.client.DiscoverTools()
	cli.prettyPrint(result, err)
}

func (cli *FinancialAnalysisCLI) Manual(symbol string) {
	tools := []string{"analyze_price", "analyze_fundamentals", "generate_company_profile"}
	params := map[string]interface{}{
		"symbol":     symbol,
		"start_date": "2024-01-01",
		"end_date":   "2024-12-31",
	}
	result, err := cli.client.ExecuteTools(tools, params, "")
	cli.prettyPrint(result, err)
}

// ---------------- EXAMPLE-LIKE COMMANDS ---------------- //

func (cli *FinancialAnalysisCLI) HealthCheck() {
	fmt.Println("=== Basic Health Check Example ===")

	// Health check
	fmt.Println("Checking server health...")
	_, err := cli.client.HealthCheck()
	if err != nil {
		fmt.Println("✗ Server is not responding")
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Println("✓ Server is healthy")

	// Server info
	infoResult, err := cli.client.GetServerInfo()
	if err != nil {
		fmt.Printf("Error getting server info: %v\n", err)
		return
	}

	if name, ok := infoResult["name"]; ok {
		fmt.Printf("Server: %v\n", name)
	}
	if version, ok := infoResult["version"]; ok {
		fmt.Printf("Version: %v\n", version)
	}
	if capabilities, ok := infoResult["capabilities"]; ok {
		fmt.Printf("Capabilities: %v\n", capabilities)
	}
}

func (cli *FinancialAnalysisCLI) ListTools() {
	fmt.Println("\n=== List Available Tools Example ===")

	result, err := cli.client.ListMCPTools()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if tools, ok := result["tools"].([]interface{}); ok {
		fmt.Printf("Available tools: %d\n", len(tools))
		for _, tool := range tools {
			if toolMap, ok := tool.(map[string]interface{}); ok {
				name := toolMap["name"]
				description := toolMap["description"]
				fmt.Printf("- %v: %v\n", name, description)
			}
		}
	}
}

func (cli *FinancialAnalysisCLI) DateUtilities() {
	fmt.Println("\n=== Date Utilities Example ===")
	now := time.Now()
	fmt.Printf("Today (date): %s\n", now.Format("2006-01-02"))
	fmt.Printf("Now (datetime): %s\n", now.Format("2006-01-02 15:04:05"))
}

// ---------------- MAIN DISPATCHER ---------------- //

func (cli *FinancialAnalysisCLI) printUsage() {
	fmt.Println("Usage: fincli <command> [args...]")
	fmt.Println("\nAvailable commands:")
	fmt.Println("  health                                    - Check server health")
	fmt.Println("  server_info                              - Get server info")
	fmt.Println("  examples                                 - Get server examples")
	fmt.Println("  analyze_price <symbol> <start> <end>     - Analyze stock price")
	fmt.Println("  analyze_sentiment <symbol> <start> <end> - Analyze sentiment")
	fmt.Println("  analyze_fundamentals <symbol> <quarters> - Analyze fundamentals")
	fmt.Println("  analyze_macro <symbol>                   - Analyze macroeconomic factors")
	fmt.Println("  company_profile <symbol>                 - Generate company profile")
	fmt.Println("  regulatory_compliance <symbol>           - Analyze regulatory compliance")
	fmt.Println("  comprehensive <symbol> <start> <end>     - Run comprehensive analysis")
	fmt.Println("  due_diligence <symbol>                   - Run due diligence analysis")
	fmt.Println("  sentiment_analysis <symbol> <start> <end> - Run sentiment-focused analysis")
	fmt.Println("  tools                                    - List all available MCP tools")
	fmt.Println("  discover_tools                           - Discover tools")
	fmt.Println("  manual <symbol>                          - Run manual multi-tool execution")
	fmt.Println("  health_check                             - Basic health check example")
	fmt.Println("  list_tools                               - List available tools example")
	fmt.Println("  date_utilities                           - Show date formatting utilities")
}

func (cli *FinancialAnalysisCLI) Run(args []string) {
	if len(args) == 0 {
		cli.printUsage()
		return
	}

	command := args[0]

	// Convert command to method name (snake_case to PascalCase)
	methodName := toPascalCase(command)

	// Get the method using reflection
	method := reflect.ValueOf(cli).MethodByName(methodName)
	if !method.IsValid() {
		fmt.Printf("Unknown command: %s\n", command)
		cli.printUsage()
		return
	}

	// Get method type info
	methodType := method.Type()
	numArgs := methodType.NumIn()

	// Check if we have the right number of arguments
	providedArgs := args[1:]
	if len(providedArgs) != numArgs {
		fmt.Printf("Command '%s' expects %d arguments, got %d\n", command, numArgs, len(providedArgs))
		return
	}

	// Convert arguments to reflect.Value slice
	callArgs := make([]reflect.Value, len(providedArgs))
	for i, arg := range providedArgs {
		callArgs[i] = reflect.ValueOf(arg)
	}

	// Call the method
	method.Call(callArgs)
}

// toPascalCase converts snake_case to PascalCase
func toPascalCase(s string) string {
	parts := strings.Split(s, "_")
	result := ""
	for _, part := range parts {
		if len(part) > 0 {
			result += strings.ToUpper(part[:1]) + strings.ToLower(part[1:])
		}
	}
	return result
}

func main() {
	cli := NewCLI()
	cli.Run(os.Args[1:])
}
