import json
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import requests


class FinancialAnalysisClientError(Exception):
    """Custom exception for Financial Analysis client errors."""

    pass


class FinancialAnalysisClient:
    """
    Comprehensive client for the Financial Analysis Server Architecture.
    Supports both Python FastAPI (port 8000) and Go Server (port 8080) endpoints.
    """

    def __init__(self, base_url="http://localhost:8080", timeout=120, session_id=None):
        """
        Initialize the client.

        Args:
            base_url: Server URL (default: Go server on port 8080)
            timeout: Request timeout in seconds
            session_id: Optional session ID for tracking
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id = session_id or f"client_session_{int(time.time())}"
        self.session = requests.Session()

        # Set session header if provided
        if self.session_id:
            self.session.headers.update({"X-Session-ID": self.session_id})

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[Any, Any]:
        """Make HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method, url=url, timeout=self.timeout, **kwargs
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise FinancialAnalysisClientError(f"Request failed: {e}")
        except json.JSONDecodeError as e:
            raise FinancialAnalysisClientError(f"Invalid JSON response: {e}")

    # ==================== Health & Info Endpoints ====================

    def health_check(self) -> Dict[str, Any]:
        """Check server health status."""
        return self._make_request("GET", "/health")

    def is_healthy(self) -> bool:
        """Simple health check that returns True/False."""
        try:
            health = self.health_check()
            return health.get("status") in ["healthy", "degraded"]
        except FinancialAnalysisClientError:
            return False

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information and capabilities."""
        return self._make_request("GET", "/info")

    def get_examples(self) -> Dict[str, Any]:
        """Get API usage examples."""
        return self._make_request("GET", "/examples")

    # ==================== MCP Protocol Methods ====================

    def list_mcp_tools(self) -> Dict[str, Any]:
        """List all available MCP tools with their schemas."""
        return self._make_request("GET", "/mcp/tools")

    def call_mcp_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a specific MCP tool directly.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
        """
        payload = {"name": tool_name, "arguments": arguments}
        return self._make_request("POST", "/mcp/call", json=payload)

    # ==================== Financial Analysis Tools ====================

    def analyze_price(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze historical price data and technical indicators.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """

        arguments = {"symbol": symbol.upper()}
        if start_date:
            arguments["start_date"] = start_date
        if end_date:
            arguments["end_date"] = end_date

        return self.call_mcp_tool("analyze_price", arguments)

    def analyze_sentiment(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze news sentiment for a stock.

        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            sources: News sources to analyze
        """
        arguments = {"symbol": symbol.upper()}
        if start_date:
            arguments["start_date"] = start_date
        if end_date:
            arguments["end_date"] = end_date
        if sources:
            arguments["sources"] = sources

        return self.call_mcp_tool("analyze_sentiment", arguments)

    def analyze_fundamentals(
        self,
        symbol: str,
        include_balance_sheet: bool = True,
        include_cash_flow: bool = True,
        include_earnings: bool = True,
        quarters: int = 4,
    ) -> Dict[str, Any]:
        """
        Analyze company fundamentals and financial statements.

        Args:
            symbol: Stock symbol
            include_balance_sheet: Include balance sheet analysis
            include_cash_flow: Include cash flow analysis
            include_earnings: Include earnings analysis
            quarters: Number of quarters to analyze
        """
        arguments = {
            "symbol": symbol.upper(),
            "include_balance_sheet": include_balance_sheet,
            "include_cash_flow": include_cash_flow,
            "include_earnings": include_earnings,
            "quarters": quarters,
        }
        return self.call_mcp_tool("analyze_fundamentals", arguments)

    def analyze_macro(
        self,
        symbol: str,
        include_sector_analysis: bool = True,
        include_economic_indicators: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze macroeconomic factors affecting a stock.

        Args:
            symbol: Stock symbol
            include_sector_analysis: Include sector-wide analysis
            include_economic_indicators: Include economic indicators
        """
        arguments = {
            "symbol": symbol.upper(),
            "include_sector_analysis": include_sector_analysis,
            "include_economic_indicators": include_economic_indicators,
        }
        return self.call_mcp_tool("analyze_macro", arguments)

    def generate_company_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Generate comprehensive company profile.

        Args:
            symbol: Stock symbol
        """
        arguments = {"symbol": symbol.upper()}
        return self.call_mcp_tool("generate_company_profile", arguments)

    def analyze_regulatory_compliance(
        self,
        symbol: str,
        include_ai_regulations: bool = True,
        include_sector_regulations: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze regulatory compliance requirements.

        Args:
            symbol: Stock symbol
            include_ai_regulations: Include AI-specific regulations
            include_sector_regulations: Include sector-specific regulations
        """
        arguments = {
            "symbol": symbol.upper(),
            "include_ai_regulations": include_ai_regulations,
            "include_sector_regulations": include_sector_regulations,
        }
        return self.call_mcp_tool("analyze_regulatory_compliance", arguments)

    # ==================== Smart Query Analysis ====================

    def query_analyze(
        self, query: str, execution_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Smart query analysis with detailed reports (recommended endpoint).

        Args:
            query: Natural language query
            execution_mode: 'parallel' or 'sequential'
        """
        payload = {"query": query}
        if execution_mode:
            payload["execution_mode"] = execution_mode

        return self._make_request("POST", "/query/analyze", json=payload)

    def api_analyze(
        self, query: str, execution_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Legacy analyze endpoint with simpler response format.

        Args:
            query: Natural language query
            execution_mode: 'parallel' or 'sequential'
        """
        payload = {"query": query}
        if execution_mode:
            payload["execution_mode"] = execution_mode

        return self._make_request("POST", "/api/analyze", json=payload)

    # ==================== Manual Tool Execution ====================

    def execute_tools(
        self,
        tools: List[str],
        parameters: Dict[str, Any],
        execution_mode: str = "parallel",
    ) -> Dict[str, Any]:
        """
        Execute multiple tools with specific parameters.

        Args:
            tools: List of tool names to execute
            parameters: Parameters to pass to all tools
            execution_mode: 'parallel' or 'sequential'
        """
        payload = {
            "tools": tools,
            "parameters": parameters,
            "execution_mode": execution_mode,
        }
        return self._make_request("POST", "/tools/execute", json=payload)

    def discover_tools(self) -> Dict[str, Any]:
        """Discover all available tools."""
        return self._make_request("GET", "/api/tools/discover")

    # ==================== Convenience Methods ====================

    def comprehensive_analysis(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        execution_mode: str = "parallel",
    ) -> Dict[str, Any]:
        """
        Run a comprehensive analysis using smart query.

        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            execution_mode: 'parallel' or 'sequential'
        """
        query = f"Give me a comprehensive analysis of {symbol.upper()}"
        if start_date and end_date:
            query += f" from {start_date} to {end_date}"
        elif end_date:
            query += f" up to {end_date}"

        return self.query_analyze(query, execution_mode)

    def due_diligence_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Run a step-by-step due diligence analysis.

        Args:
            symbol: Stock symbol
        """
        query = f"Perform step by step due diligence analysis for {symbol.upper()}"
        return self.query_analyze(query, "sequential")

    def sentiment_analysis(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Focus on sentiment and news analysis.

        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        query = f"Analyze sentiment and news for {symbol.upper()}"
        if start_date and end_date:
            query += f" from {start_date} to {end_date}"

        return self.query_analyze(query)

    # ==================== Utility Methods ====================

    @staticmethod
    def format_date(date_obj) -> str:
        """
        Format date object to YYYY-MM-DD string.

        Args:
            date_obj: datetime.date, datetime.datetime, or string
        """
        if isinstance(date_obj, (datetime, date)):
            return date_obj.strftime("%Y-%m-%d")
        return str(date_obj)

    def close(self):
        """Close the session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
