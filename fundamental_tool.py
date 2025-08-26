# ============================================================================
# Fundamental Analysis
# ============================================================================

import requests
from typing import Dict, Any, Optional, List
import dspy
from pydantic import PrivateAttr

from base_tool import BaseFinancialTool
from utility import retry_with_backoff, cache_result
from config import Config
from log import LoggerManager


class FundamentalAnalysisSignature(dspy.Signature):
    """Signature for fundamental analysis."""

    symbol = dspy.InputField(desc="Stock symbol")
    financial_data = dspy.InputField(desc="Formatted financial data")
    analysis = dspy.OutputField(desc="Fundamental analysis")


class FundamentalAnalyzer(BaseFinancialTool):
    """Analyze fundamental data using Alpha Vantage."""

    name: str = "FundamentalAnalyzer"
    input_type: type = str
    desc: str = "Analyzes company fundamentals including financials and earnings using Alpha Vantage"
    tool: Optional[str] = None

    # Declare private attributes for Pydantic compatibility
    _analyze: Any = PrivateAttr()
    _base_url: str = PrivateAttr()
    _api_available: bool = PrivateAttr()

    def __init__(
        self, config: Optional[Config] = None, logger: Optional[LoggerManager] = None
    ):
        super().__init__(config, logger)
        self._analyze = dspy.ChainOfThought(FundamentalAnalysisSignature)
        self._base_url = "https://www.alphavantage.co/query"

        # Check for API key availability
        if not self._config.alpha_vantage_key:
            self._logger.warning(
                "Alpha Vantage API key not found - fundamental analysis will be limited"
            )
            self._api_available = False
        else:
            self._api_available = True

    @retry_with_backoff(
        max_retries=2, retry_delay=15.0, logger=LoggerManager()
    )  # Alpha Vantage rate limit
    @cache_result(cache_dir=".cache", logger=LoggerManager())
    def fetch_income_statement(self, symbol: str) -> Dict[str, Any]:
        """Fetch income statement data from Alpha Vantage."""
        if not self._api_available:
            return {}

        try:
            self._logger.info(f"Fetching income statement for {symbol}")

            params = {
                "function": "INCOME_STATEMENT",
                "symbol": symbol,
                "apikey": self._config.alpha_vantage_key,
            }

            response = requests.get(self._base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                self._logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return {}

            if "Note" in data:
                self._logger.warning(f"Alpha Vantage rate limit note: {data['Note']}")
                return {}

            return {
                "quarterly_reports": data.get("quarterlyReports", []),
                "annual_reports": data.get("annualReports", []),
            }

        except requests.RequestException as e:
            self._logger.error(
                f"Network error fetching income statement for {symbol}: {e}"
            )
            raise
        except Exception as e:
            self._logger.error(f"Error fetching income statement for {symbol}: {e}")
            raise

    @retry_with_backoff(max_retries=2, retry_delay=15.0, logger=LoggerManager())
    @cache_result(cache_dir=".cache", logger=LoggerManager())
    def fetch_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """Fetch balance sheet data from Alpha Vantage."""
        if not self._api_available:
            return {}

        try:
            self._logger.info(f"Fetching balance sheet for {symbol}")

            params = {
                "function": "BALANCE_SHEET",
                "symbol": symbol,
                "apikey": self._config.alpha_vantage_key,
            }

            response = requests.get(self._base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data or "Note" in data:
                return {}

            return {
                "quarterly_reports": data.get("quarterlyReports", []),
                "annual_reports": data.get("annualReports", []),
            }

        except Exception as e:
            self._logger.error(f"Error fetching balance sheet for {symbol}: {e}")
            return {}

    @retry_with_backoff(max_retries=2, retry_delay=15.0, logger=LoggerManager())
    @cache_result(cache_dir=".cache", logger=LoggerManager())
    def fetch_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """Fetch cash flow statement data from Alpha Vantage."""
        if not self._api_available:
            return {}

        try:
            self._logger.info(f"Fetching cash flow statement for {symbol}")

            params = {
                "function": "CASH_FLOW",
                "symbol": symbol,
                "apikey": self._config.alpha_vantage_key,
            }

            response = requests.get(self._base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data or "Note" in data:
                return {}

            return {
                "quarterly_reports": data.get("quarterlyReports", []),
                "annual_reports": data.get("annualReports", []),
            }

        except Exception as e:
            self._logger.error(f"Error fetching cash flow for {symbol}: {e}")
            return {}

    @retry_with_backoff(max_retries=2, retry_delay=15.0, logger=LoggerManager())
    @cache_result(cache_dir=".cache", logger=LoggerManager())
    def fetch_earnings(self, symbol: str) -> Dict[str, Any]:
        """Fetch earnings data from Alpha Vantage."""
        if not self._api_available:
            return {}

        try:
            self._logger.info(f"Fetching earnings for {symbol}")

            params = {
                "function": "EARNINGS",
                "symbol": symbol,
                "apikey": self._config.alpha_vantage_key,
            }

            response = requests.get(self._base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data or "Note" in data:
                return {}

            return {
                "quarterly_earnings": data.get("quarterlyEarnings", []),
                "annual_earnings": data.get("annualEarnings", []),
            }

        except Exception as e:
            self._logger.error(f"Error fetching earnings for {symbol}: {e}")
            return {}

    def fetch_fundamentals(
        self,
        symbol: str,
        include_balance_sheet: bool = False,
        include_cash_flow: bool = False,
        include_earnings: bool = False,
    ) -> Dict[str, Any]:
        """Fetch comprehensive fundamental data."""
        fundamentals = {}

        # Always fetch income statement
        income_data = self.fetch_income_statement(symbol)
        if income_data:
            fundamentals.update(income_data)

        # Optionally fetch balance sheet
        if include_balance_sheet:
            balance_data = self.fetch_balance_sheet(symbol)
            if balance_data:
                fundamentals["balance_sheet_quarterly"] = balance_data.get(
                    "quarterly_reports", []
                )
                fundamentals["balance_sheet_annual"] = balance_data.get(
                    "annual_reports", []
                )

        # Optionally fetch cash flow
        if include_cash_flow:
            cash_flow_data = self.fetch_cash_flow(symbol)
            if cash_flow_data:
                fundamentals["cash_flow_quarterly"] = cash_flow_data.get(
                    "quarterly_reports", []
                )
                fundamentals["cash_flow_annual"] = cash_flow_data.get(
                    "annual_reports", []
                )

        # Optionally fetch earnings
        if include_earnings:
            earnings_data = self.fetch_earnings(symbol)
            if earnings_data:
                fundamentals["earnings_quarterly"] = earnings_data.get(
                    "quarterly_earnings", []
                )
                fundamentals["earnings_annual"] = earnings_data.get(
                    "annual_earnings", []
                )

        return fundamentals

    def format_financial_metrics(self, report: Dict[str, Any]) -> Dict[str, float]:
        """Extract and format key financial metrics from a report."""
        metrics = {}

        # Revenue metrics
        total_revenue = report.get("totalRevenue", "0")
        metrics["revenue"] = float(total_revenue) if total_revenue != "None" else 0

        # Profitability metrics
        net_income = report.get("netIncome", "0")
        metrics["net_income"] = float(net_income) if net_income != "None" else 0

        # Per-share metrics
        reported_eps = report.get("reportedEPS", "0")
        metrics["eps"] = float(reported_eps) if reported_eps != "None" else 0

        # Operating metrics
        operating_income = report.get("operatingIncome", "0")
        metrics["operating_income"] = (
            float(operating_income) if operating_income != "None" else 0
        )

        # Calculate margins if possible
        if metrics["revenue"] > 0:
            metrics["net_margin"] = (metrics["net_income"] / metrics["revenue"]) * 100
            metrics["operating_margin"] = (
                metrics["operating_income"] / metrics["revenue"]
            ) * 100

        return metrics

    def format_cash_flow_metrics(self, report: Dict[str, Any]) -> Dict[str, float]:
        """Extract and format key cash flow metrics from a report."""
        metrics = {}

        # Operating cash flow
        operating_cash_flow = report.get("operatingCashflow", "0")
        metrics["operating_cash_flow"] = (
            float(operating_cash_flow) if operating_cash_flow != "None" else 0
        )

        # Investing cash flow
        investing_cash_flow = report.get("cashflowFromInvestment", "0")
        metrics["investing_cash_flow"] = (
            float(investing_cash_flow) if investing_cash_flow != "None" else 0
        )

        # Financing cash flow
        financing_cash_flow = report.get("cashflowFromFinancing", "0")
        metrics["financing_cash_flow"] = (
            float(financing_cash_flow) if financing_cash_flow != "None" else 0
        )

        # Free cash flow (Operating CF - Capital Expenditures)
        capital_expenditures = report.get("capitalExpenditures", "0")
        capex = float(capital_expenditures) if capital_expenditures != "None" else 0
        metrics["free_cash_flow"] = metrics["operating_cash_flow"] - abs(
            capex
        )  # CapEx is usually negative

        # Net change in cash
        net_change_cash = report.get("changeInCashAndCashEquivalents", "0")
        metrics["net_change_cash"] = (
            float(net_change_cash) if net_change_cash != "None" else 0
        )

        return metrics

    def format_earnings_metrics(self, report: Dict[str, Any]) -> Dict[str, float]:
        """Extract and format key earnings metrics from a report."""
        metrics = {}

        # Reported EPS
        reported_eps = report.get("reportedEPS", "0")
        metrics["reported_eps"] = (
            float(reported_eps) if reported_eps not in ["None", None, ""] else 0
        )

        # Estimated EPS (for comparison)
        estimated_eps = report.get("estimatedEPS", "0")
        metrics["estimated_eps"] = (
            float(estimated_eps) if estimated_eps not in ["None", None, ""] else 0
        )

        # Surprise (difference between reported and estimated)
        if metrics["estimated_eps"] != 0:
            metrics["eps_surprise"] = metrics["reported_eps"] - metrics["estimated_eps"]
            metrics["eps_surprise_percent"] = (
                metrics["eps_surprise"] / abs(metrics["estimated_eps"])
            ) * 100
        else:
            metrics["eps_surprise"] = 0
            metrics["eps_surprise_percent"] = 0

        # Surprise percentage (as provided by API)
        surprise_percent = report.get("surprisePercentage", "0")
        if surprise_percent not in ["None", None, ""]:
            metrics["surprise_percentage"] = float(surprise_percent)

        return metrics

    def format_fundamentals(self, data: Dict[str, Any]) -> str:
        """Format financial data for analysis."""
        if not data:
            return "No fundamental data available"

        formatted = []

        # Format quarterly data (Income Statement)
        quarterly_reports = data.get("quarterly_reports", [])
        if quarterly_reports:
            latest_quarter = quarterly_reports[0]
            metrics = self.format_financial_metrics(latest_quarter)

            formatted.append(
                f"Latest Quarter ({latest_quarter.get('fiscalDateEnding', 'N/A')}):"
            )
            formatted.append(f"  Revenue: ${metrics['revenue']:,.0f}")
            formatted.append(f"  Net Income: ${metrics['net_income']:,.0f}")
            formatted.append(f"  EPS: ${metrics['eps']:.2f}")

            if metrics.get("net_margin") is not None:
                formatted.append(f"  Net Margin: {metrics['net_margin']:.1f}%")
            if metrics.get("operating_margin") is not None:
                formatted.append(
                    f"  Operating Margin: {metrics['operating_margin']:.1f}%"
                )

            # Quarter-over-quarter comparison if available
            if len(quarterly_reports) > 1:
                prev_quarter = quarterly_reports[1]
                prev_metrics = self.format_financial_metrics(prev_quarter)

                if prev_metrics["revenue"] > 0:
                    revenue_growth = (
                        (metrics["revenue"] - prev_metrics["revenue"])
                        / prev_metrics["revenue"]
                    ) * 100
                    formatted.append(f"  QoQ Revenue Growth: {revenue_growth:.1f}%")

        # Format annual data (Income Statement)
        annual_reports = data.get("annual_reports", [])
        if annual_reports:
            latest_annual = annual_reports[0]
            annual_metrics = self.format_financial_metrics(latest_annual)

            formatted.append(
                f"\nLatest Annual ({latest_annual.get('fiscalDateEnding', 'N/A')}):"
            )
            formatted.append(f"  Revenue: ${annual_metrics['revenue']:,.0f}")
            formatted.append(f"  Net Income: ${annual_metrics['net_income']:,.0f}")

            if annual_metrics.get("net_margin") is not None:
                formatted.append(f"  Net Margin: {annual_metrics['net_margin']:.1f}%")

            # Year-over-year comparison if available
            if len(annual_reports) > 1:
                prev_annual = annual_reports[1]
                prev_annual_metrics = self.format_financial_metrics(prev_annual)

                if prev_annual_metrics["revenue"] > 0:
                    yoy_growth = (
                        (annual_metrics["revenue"] - prev_annual_metrics["revenue"])
                        / prev_annual_metrics["revenue"]
                    ) * 100
                    formatted.append(f"  YoY Revenue Growth: {yoy_growth:.1f}%")

        # Format Cash Flow data
        cash_flow_quarterly = data.get("cash_flow_quarterly", [])
        if cash_flow_quarterly:
            latest_cf = cash_flow_quarterly[0]
            cf_metrics = self.format_cash_flow_metrics(latest_cf)

            formatted.append(
                f"\nLatest Quarter Cash Flow ({latest_cf.get('fiscalDateEnding', 'N/A')}):"
            )
            formatted.append(
                f"  Operating Cash Flow: ${cf_metrics['operating_cash_flow']:,.0f}"
            )
            formatted.append(f"  Free Cash Flow: ${cf_metrics['free_cash_flow']:,.0f}")
            formatted.append(
                f"  Investing Cash Flow: ${cf_metrics['investing_cash_flow']:,.0f}"
            )
            formatted.append(
                f"  Financing Cash Flow: ${cf_metrics['financing_cash_flow']:,.0f}"
            )

        # Format annual cash flow
        cash_flow_annual = data.get("cash_flow_annual", [])
        if cash_flow_annual:
            latest_annual_cf = cash_flow_annual[0]
            annual_cf_metrics = self.format_cash_flow_metrics(latest_annual_cf)

            formatted.append(
                f"\nLatest Annual Cash Flow ({latest_annual_cf.get('fiscalDateEnding', 'N/A')}):"
            )
            formatted.append(
                f"  Operating Cash Flow: ${annual_cf_metrics['operating_cash_flow']:,.0f}"
            )
            formatted.append(
                f"  Free Cash Flow: ${annual_cf_metrics['free_cash_flow']:,.0f}"
            )

        # Format Earnings data
        earnings_quarterly = data.get("earnings_quarterly", [])
        if earnings_quarterly:
            latest_earnings = earnings_quarterly[0]
            earnings_metrics = self.format_earnings_metrics(latest_earnings)

            formatted.append(
                f"\nLatest Quarter Earnings ({latest_earnings.get('fiscalDateEnding', 'N/A')}):"
            )
            formatted.append(f"  Reported EPS: ${earnings_metrics['reported_eps']:.2f}")
            formatted.append(
                f"  Estimated EPS: ${earnings_metrics['estimated_eps']:.2f}"
            )

            if earnings_metrics.get("surprise_percentage") is not None:
                formatted.append(
                    f"  Surprise: {earnings_metrics['surprise_percentage']:.1f}%"
                )
            elif earnings_metrics.get("eps_surprise_percent") != 0:
                formatted.append(
                    f"  EPS Surprise: {earnings_metrics['eps_surprise_percent']:+.1f}%"
                )

            # Compare last few quarters for trend
            if len(earnings_quarterly) > 1:
                formatted.append(f"  Recent EPS Trend:")
                for i, quarter in enumerate(earnings_quarterly[:4]):  # Last 4 quarters
                    q_metrics = self.format_earnings_metrics(quarter)
                    formatted.append(
                        f"    Q{4 - i}: ${q_metrics['reported_eps']:.2f} ({quarter.get('fiscalDateEnding', 'N/A')})"
                    )

        # Format annual earnings
        earnings_annual = data.get("earnings_annual", [])
        if earnings_annual:
            latest_annual_earnings = earnings_annual[0]
            annual_earnings_metrics = self.format_earnings_metrics(
                latest_annual_earnings
            )

            formatted.append(
                f"\nLatest Annual Earnings ({latest_annual_earnings.get('fiscalDateEnding', 'N/A')}):"
            )
            formatted.append(
                f"  Annual EPS: ${annual_earnings_metrics['reported_eps']:.2f}"
            )

            # Year-over-year EPS growth
            if len(earnings_annual) > 1:
                prev_annual_earnings = earnings_annual[1]
                prev_earnings_metrics = self.format_earnings_metrics(
                    prev_annual_earnings
                )

                if prev_earnings_metrics["reported_eps"] != 0:
                    eps_growth = (
                        (
                            annual_earnings_metrics["reported_eps"]
                            - prev_earnings_metrics["reported_eps"]
                        )
                        / abs(prev_earnings_metrics["reported_eps"])
                    ) * 100
                    formatted.append(f"  YoY EPS Growth: {eps_growth:+.1f}%")

        return "\n".join(formatted) if formatted else "Limited financial data available"

    def forward(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Process the fundamental analysis request."""
        # Validate inputs using base class method
        if not self.validate_inputs(symbol):
            return {
                "symbol": symbol,
                "error": f"Invalid symbol provided: {symbol}",
                "analysis": "Analysis unavailable due to invalid input",
                "status": "invalid_input",
            }

        # Check API availability
        if not self._api_available:
            return {
                "symbol": symbol,
                "error": "Alpha Vantage API key not configured",
                "analysis": "Fundamental analysis requires Alpha Vantage API access",
                "status": "no_api_key",
            }

        try:
            # Extract optional parameters
            include_balance_sheet = kwargs.get("include_balance_sheet", False)
            include_cash_flow = kwargs.get("include_cash_flow", False)
            include_earnings = kwargs.get("include_earnings", False)

            # Fetch fundamental data
            data = self.fetch_fundamentals(
                symbol,
                include_balance_sheet=include_balance_sheet,
                include_cash_flow=include_cash_flow,
                include_earnings=include_earnings,
            )

            if not data:
                return {
                    "symbol": symbol,
                    "analysis": f"No fundamental data available for {symbol}",
                    "data_sources": [],
                    "status": "no_data",
                }

            # Format data for analysis
            formatted_data = self.format_fundamentals(data)

            # Perform fundamental analysis
            analysis_result = self._analyze(
                symbol=symbol, financial_data=formatted_data
            )

            # Determine available data sources
            data_sources = []
            if data.get("quarterly_reports"):
                data_sources.append("quarterly_income_statement")
            if data.get("annual_reports"):
                data_sources.append("annual_income_statement")
            if data.get("balance_sheet_quarterly"):
                data_sources.append("quarterly_balance_sheet")
            if data.get("balance_sheet_annual"):
                data_sources.append("annual_balance_sheet")
            if data.get("cash_flow_quarterly"):
                data_sources.append("quarterly_cash_flow")
            if data.get("cash_flow_annual"):
                data_sources.append("annual_cash_flow")
            if data.get("earnings_quarterly"):
                data_sources.append("quarterly_earnings")
            if data.get("earnings_annual"):
                data_sources.append("annual_earnings")

            return {
                "symbol": symbol,
                "analysis": analysis_result.analysis,
                "data_sources": data_sources,
                "quarters_available": len(data.get("quarterly_reports", [])),
                "years_available": len(data.get("annual_reports", [])),
                "earnings_quarters_available": len(data.get("earnings_quarterly", [])),
                "earnings_years_available": len(data.get("earnings_annual", [])),
                "formatted_data": formatted_data,
                "status": "success",
            }

        except Exception as e:
            self._logger.error(f"Fundamental analysis error for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "analysis": f"Unable to complete fundamental analysis: {str(e)}",
                "status": "error",
            }

    def validate_inputs(self, symbol: str) -> bool:
        """Enhanced input validation for fundamental analysis."""
        # Use base class validation first
        if not super().validate_inputs(symbol):
            return False

        # Additional validation specific to fundamental analysis
        if len(symbol) > 10:
            self._logger.warning(f"Symbol too long: {symbol}")
            return False

        return True


if __name__ == "__main__":
    # Initialize the tool
    from config import Config

    import os
    from dotenv import load_dotenv

    load_dotenv()
    chatbot_key = os.getenv("CHATBOT_KEY")

    config = Config()

    config_custom = Config(
        model_name="openrouter/deepseek/deepseek-chat-v3-0324:free",
        api_base="https://openrouter.ai/api/v1",
        api_key=chatbot_key,
        llm_temperature=0.0,
    )

    # Set up the language model
    lm = config_custom.setup_dspy_lm()
    dspy.settings.configure(lm=lm)

    # Basic usage
    analyzer = FundamentalAnalyzer(config_custom)

    # Comprehensive analysis with all data
    result = analyzer.forward(
        "MSFT",
        include_balance_sheet=True,
        include_cash_flow=True,
        include_earnings=True,
    )

    print("Analysis Result:")
    print("________________________________________________")
    print(f"Impact: {result['status']}")
    print("________________________________________________")
    print(f"Symbol: {result['symbol']}")
    print("________________________________________________")
    print(f"Status: {result['analysis']}")
    print("________________________________________________")
    print(f"Outlook: {result['formatted_data']}")
