import sys
from datetime import date, datetime

from client import FinancialAnalysisClient, FinancialAnalysisClientError


class FinancialAnalysisCLI:
    """Command-line interface for FinancialAnalysisClient."""

    def __init__(self):
        self.client = FinancialAnalysisClient()

    # ---------------- BASIC HEALTH & INFO ---------------- #

    def health(self):
        """Check server health."""
        if self.client.is_healthy():
            print("✅ Server is healthy")
        else:
            print("❌ Server is not healthy")

    def server_info(self):
        """Get server info."""
        try:
            info = self.client.get_server_info()
            print(info)
        except FinancialAnalysisClientError as e:
            print(f"Error: {e}")

    # ---------------- ANALYSIS COMMANDS ---------------- #

    def analyze_price(self, symbol, start_date=None, end_date=None):
        """Analyze stock price."""
        result = self.client.analyze_price(symbol, start_date, end_date)
        print(result)

    def analyze_sentiment(self, symbol, start_date=None, end_date=None):
        """Analyze sentiment."""
        result = self.client.analyze_sentiment(symbol, start_date, end_date)
        print(result)

    def analyze_fundamentals(self, symbol, quarters="4"):
        """Analyze fundamentals."""
        result = self.client.analyze_fundamentals(symbol, quarters=int(quarters))
        print(result)

    def analyze_macro(self, symbol):
        """Analyze macroeconomic factors."""
        result = self.client.analyze_macro(symbol)
        print(result)

    def company_profile(self, symbol):
        """Generate company profile."""
        result = self.client.generate_company_profile(symbol)
        print(result)

    def regulatory_compliance(self, symbol):
        """Analyze regulatory compliance."""
        result = self.client.analyze_regulatory_compliance(symbol)
        print(result)

    # ---------------- QUERY / COMPOSITE COMMANDS ---------------- #

    def comprehensive(self, symbol, start_date=None, end_date=None):
        """Run comprehensive analysis."""
        result = self.client.comprehensive_analysis(symbol, start_date, end_date)
        print(result)

    def due_diligence(self, symbol):
        """Run due diligence analysis."""
        result = self.client.due_diligence_analysis(symbol)
        print(result)

    def sentiment_analysis(self, symbol, start_date=None, end_date=None):
        """Run sentiment-focused analysis."""
        result = self.client.sentiment_analysis(symbol, start_date, end_date)
        print(result)

    # ---------------- TOOLS ---------------- #

    def tools(self):
        """List all available MCP tools."""
        result = self.client.list_mcp_tools()
        print(result)

    def discover_tools(self):
        """Discover tools."""
        result = self.client.discover_tools()
        print(result)

    def manual(self, symbol):
        """Run manual multi-tool execution."""
        tools = ["analyze_price", "analyze_fundamentals", "generate_company_profile"]
        params = {
            "symbol": symbol,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        }
        result = self.client.execute_tools(tools, params)
        print(result)

    # ---------------- EXAMPLE-LIKE COMMANDS ---------------- #

    def health_check(self):
        """Basic health check and server info example."""
        print("=== Basic Health Check Example ===")
        with FinancialAnalysisClient() as client:
            try:
                print("Checking server health...")
                if client.is_healthy():
                    print("✓ Server is healthy")
                    info = client.get_server_info()
                    print(f"Server: {info.get('name', 'Unknown')}")
                    print(f"Version: {info.get('version', 'Unknown')}")
                    print(f"Capabilities: {info.get('capabilities', [])}")
                else:
                    print("✗ Server is not responding")
            except FinancialAnalysisClientError as e:
                print(f"Error: {e}")

    def list_tools(self):
        """List available tools example."""
        print("\n=== List Available Tools Example ===")
        with FinancialAnalysisClient() as client:
            try:
                tools = client.list_mcp_tools()
                print(f"Available tools: {len(tools.get('tools', []))}")
                for tool in tools.get("tools", []):
                    print(f"- {tool['name']}: {tool['description']}")
            except FinancialAnalysisClientError as e:
                print(f"Error: {e}")

    def date_utilities(self):
        """Show date formatting utilities."""
        print("\n=== Date Utilities Example ===")
        today = date.today()
        now = datetime.now()
        print(f"Today (date): {FinancialAnalysisClient.format_date(today)}")
        print(f"Now (datetime): {FinancialAnalysisClient.format_date(now)}")

    # ---------------- MAIN DISPATCHER ---------------- #

    def run(self, args):
        if not args:
            print("Usage: python cli.py <command> [args...]")
            return

        command = args[0]
        method = getattr(self, command, None)

        if not method:
            print(f"Unknown command: {command}")
            return

        try:
            method(*args[1:])
        except TypeError as e:
            print(f"Invalid arguments for '{command}': {e}")
        except FinancialAnalysisClientError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    cli = FinancialAnalysisCLI()
    cli.run(sys.argv[1:])
