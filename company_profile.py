from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import dspy
import requests
import yfinance as yf

from config import Config

# Import our logging and config systems
from log import LoggerManager


@dataclass
class CompanyProfile:
    """Structured output for company profile"""

    ticker: str
    name: str
    description: str
    sector: str
    industry: str
    market_cap: Optional[str]
    pe_ratio: Optional[float]
    dividend_yield: Optional[float]
    current_price: Optional[float]
    price_change: Optional[str]
    key_metrics: Dict[str, Any]
    recent_news_sentiment: Optional[str]
    business_highlights: list
    risks: list
    financial_summary: str
    generated_at: str


class CompanyDescriptionSignature(dspy.Signature):
    """Generate comprehensive company description and analysis"""

    company_name = dspy.InputField(desc="Company name or ticker symbol")
    sector = dspy.InputField(desc="Company sector/industry")
    market_data = dspy.InputField(desc="Current market data and financials")

    description = dspy.OutputField(
        desc="Comprehensive company description covering business model, operations, and market position"
    )
    business_highlights = dspy.OutputField(
        desc="Key business strengths and competitive advantages as bullet points"
    )
    risks = dspy.OutputField(desc="Main business risks and challenges as bullet points")
    financial_summary = dspy.OutputField(
        desc="Analysis of financial health and performance trends"
    )


class CompanyProfileGenerator(dspy.Module):
    """DSPy module for generating comprehensive company profiles"""

    def __init__(
        self,
        alphavantage_api_key: str = None,
        sentiment_tool=None,
        config: Config = None,
        logger: LoggerManager = None,
    ):
        super().__init__()

        # Setup logging and config
        self.logger = logger or LoggerManager(name="company_profile", level="DEBUG")
        self.config = config or Config()

        # Use API key from config if not provided
        self.alphavantage_key = alphavantage_api_key or self.config.alpha_vantage_key
        self.sentiment_tool = sentiment_tool

        self.logger.info("Initializing CompanyProfileGenerator", component="init")

        if not self.alphavantage_key:
            self.logger.warning(
                "No AlphaVantage API key provided - some features may be limited",
                component="init",
            )

        self.company_analyzer = dspy.ChainOfThought(CompanyDescriptionSignature)

    def get_yfinance_description(self, ticker: str) -> str:
        """Get only the business description from yfinance"""
        try:
            self.logger.debug(
                f"Fetching yfinance description for {ticker}", component="yfinance"
            )
            stock = yf.Ticker(ticker)
            info = stock.info
            description = info.get("longBusinessSummary", "")
            if description:
                self.logger.info(
                    f"Successfully retrieved yfinance description for {ticker}",
                    component="yfinance",
                )
            else:
                self.logger.warning(
                    f"No business summary found for {ticker} in yfinance",
                    component="yfinance",
                )
            return description
        except Exception as e:
            self.logger.error(
                f"Error fetching yfinance description for {ticker}",
                error=str(e),
                component="yfinance",
            )
            return ""

    def get_alphavantage_overview(self, ticker: str) -> Dict[str, Any]:
        """Get company overview from AlphaVantage"""
        if not self.alphavantage_key:
            self.logger.warning(
                "No AlphaVantage API key available", component="alphavantage"
            )
            return {}

        try:
            self.logger.debug(
                f"Fetching AlphaVantage overview for {ticker}", component="alphavantage"
            )
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": self.alphavantage_key,
            }

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                self.logger.error(
                    f"AlphaVantage API error for {ticker}",
                    error=data["Error Message"],
                    component="alphavantage",
                )
                return {}

            if "Note" in data:
                self.logger.warning(
                    f"AlphaVantage API limit reached for {ticker}",
                    note=data["Note"],
                    component="alphavantage",
                )
                return {}

            # Parse and clean the data
            if "Symbol" in data:  # Valid response
                self.logger.info(
                    f"Successfully retrieved AlphaVantage data for {ticker}",
                    component="alphavantage",
                )
                return {
                    "name": data.get("Name", ""),
                    "sector": data.get("Sector", ""),
                    "industry": data.get("Industry", ""),
                    "market_cap": self._parse_number(
                        data.get("MarketCapitalization", "0")
                    ),
                    "pe_ratio": self._parse_float(data.get("PERatio", "None")),
                    "dividend_yield": self._parse_float(
                        data.get("DividendYield", "None")
                    ),
                    "revenue_ttm": self._parse_number(data.get("RevenueTTM", "0")),
                    "profit_margin": self._parse_float(
                        data.get("ProfitMargin", "None")
                    ),
                    "operating_margin": self._parse_float(
                        data.get("OperatingMarginTTM", "None")
                    ),
                    "roe": self._parse_float(data.get("ReturnOnEquityTTM", "None")),
                    "debt_to_equity": self._parse_float(
                        data.get("DebtToEquityRatio", "None")
                    ),
                    "current_ratio": self._parse_float(
                        data.get("CurrentRatio", "None")
                    ),
                    "book_value": self._parse_float(data.get("BookValue", "None")),
                    "eps": self._parse_float(data.get("EPS", "None")),
                    "beta": self._parse_float(data.get("Beta", "None")),
                    "52_week_high": self._parse_float(data.get("52WeekHigh", "None")),
                    "52_week_low": self._parse_float(data.get("52WeekLow", "None")),
                    "shares_outstanding": self._parse_number(
                        data.get("SharesOutstanding", "0")
                    ),
                    "description": data.get("Description", ""),
                }

            else:
                self.logger.warning(
                    f"No valid data found for {ticker} in AlphaVantage response",
                    component="alphavantage",
                )
            return {}
        except Exception as e:
            self.logger.error(
                f"Error fetching AlphaVantage overview for {ticker}",
                error=str(e),
                exc_info=True,
                component="alphavantage",
            )
            return {}

    def _parse_number(self, value: str) -> int:
        """Parse number string to int (handles None, '-', etc.)"""
        if not value or value in ["None", "-", "N/A"]:
            return 0
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    def _parse_float(self, value: str) -> Optional[float]:
        """Parse float string to float (handles None, '-', etc.)"""
        if not value or value in ["None", "-", "N/A"]:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def format_market_cap(self, market_cap: int) -> str:
        """Format market cap in readable format"""
        if market_cap >= 1e12:
            return f"${market_cap / 1e12:.2f}T"
        elif market_cap >= 1e9:
            return f"${market_cap / 1e9:.2f}B"
        elif market_cap >= 1e6:
            return f"${market_cap / 1e6:.2f}M"
        else:
            return f"${market_cap:,.0f}"

    def forward(self, company_identifier: str) -> CompanyProfile:
        """Generate comprehensive company profile"""

        # Assume input is ticker symbol (can be enhanced with ticker lookup)
        ticker = company_identifier.upper()

        self.logger.info(
            f"Starting company profile generation for {ticker}", component="forward"
        )

        # Get AlphaVantage overview data (primary source)
        self.logger.debug("Fetching AlphaVantage data", component="forward")
        av_data = self.get_alphavantage_overview(ticker)

        # Get business description from yfinance as fallback
        self.logger.debug("Fetching yfinance data", component="forward")
        yf_description = self.get_yfinance_description(ticker)

        # Use your existing sentiment tool if available
        sentiment = None
        if self.sentiment_tool:
            try:
                self.logger.debug("Analyzing sentiment", component="forward")
                sentiment = self.sentiment_tool.analyze(ticker)
                self.logger.info("Sentiment analysis completed", component="forward")
            except Exception as e:
                self.logger.error(
                    "Error with sentiment tool", error=str(e), component="forward"
                )

        # Prepare market data context for LLM
        market_context = f"""
        Market Cap: {self.format_market_cap(av_data.get("market_cap", 0)) if av_data.get("market_cap") else "N/A"}
        P/E Ratio: {av_data.get("pe_ratio") or "N/A"}
        Revenue (TTM): {f"${av_data.get('revenue_ttm', 0) / 1e9:.2f}B" if av_data.get("revenue_ttm") else "N/A"}
        Profit Margin: {f"{av_data.get('profit_margin') * 100:.1f}%" if av_data.get("profit_margin") else "N/A"}
        Operating Margin: {f"{av_data.get('operating_margin') * 100:.1f}%" if av_data.get("operating_margin") else "N/A"}
        ROE: {f"{av_data.get('roe') * 100:.1f}%" if av_data.get("roe") else "N/A"}
        Debt-to-Equity: {av_data.get("debt_to_equity") or "N/A"}
        Current Ratio: {av_data.get("current_ratio") or "N/A"}
        EPS: {f"${av_data.get('eps'):.2f}" if av_data.get("eps") else "N/A"}
        Beta: {av_data.get("beta") or "N/A"}
        52-Week Range: ${av_data.get("52_week_low") or "N/A"} - ${av_data.get("52_week_high") or "N/A"}
        Book Value: {f"${av_data.get('book_value'):.2f}" if av_data.get("book_value") else "N/A"}
        Dividend Yield: {f"{av_data.get('dividend_yield') * 100:.2f}%" if av_data.get("dividend_yield") else "N/A"}
        Recent Sentiment: {sentiment or "N/A"}
        """

        # Generate LLM analysis
        company_name = av_data.get("name") or company_identifier
        sector = av_data.get("sector") or "Unknown"

        # Combine descriptions (prioritize AlphaVantage, fallback to yfinance)
        base_description = av_data.get("description") or yf_description

        self.logger.debug(
            "Starting LLM analysis",
            company=company_name,
            sector=sector,
            component="forward",
        )

        try:
            result = self.company_analyzer(
                company_name=company_name, sector=sector, market_data=market_context
            )
            self.logger.info("LLM analysis completed successfully", component="forward")
        except Exception as e:
            self.logger.error(
                "Error during LLM analysis",
                error=str(e),
                exc_info=True,
                component="forward",
            )
            # Create fallback result
            result = type(
                "obj",
                (object,),
                {
                    "description": base_description
                    or f"Company profile for {company_name}",
                    "business_highlights": "• Data analysis in progress",
                    "risks": "• Analysis pending",
                    "financial_summary": "Financial analysis unavailable",
                },
            )()

        # Parse business highlights and risks from LLM output
        try:
            highlights = [
                h.strip("• -").strip()
                for h in result.business_highlights.split("\n")
                if h.strip()
            ]
            risks = [
                r.strip("• -").strip() for r in result.risks.split("\n") if r.strip()
            ]
            self.logger.debug(
                f"Parsed {len(highlights)} highlights and {len(risks)} risks",
                component="forward",
            )
        except Exception as e:
            self.logger.warning(
                "Error parsing highlights and risks", error=str(e), component="forward"
            )
            highlights = ["Analysis in progress"]
            risks = ["Analysis pending"]

        # Compile key metrics from AlphaVantage
        key_metrics = {
            "revenue_ttm": av_data.get("revenue_ttm"),
            "profit_margin": av_data.get("profit_margin"),
            "operating_margin": av_data.get("operating_margin"),
            "debt_to_equity": av_data.get("debt_to_equity"),
            "return_on_equity": av_data.get("roe"),
            "current_ratio": av_data.get("current_ratio"),
            "eps": av_data.get("eps"),
            "beta": av_data.get("beta"),
            "book_value": av_data.get("book_value"),
        }

        profile = CompanyProfile(
            ticker=ticker,
            name=company_name,
            description=result.description,
            sector=av_data.get("sector", "Unknown"),
            industry=av_data.get("industry", "Unknown"),
            market_cap=(
                self.format_market_cap(av_data.get("market_cap", 0))
                if av_data.get("market_cap")
                else None
            ),
            pe_ratio=av_data.get("pe_ratio"),
            dividend_yield=av_data.get("dividend_yield"),
            current_price=None,  # AlphaVantage OVERVIEW doesn't include current price
            price_change=None,  # Would need separate GLOBAL_QUOTE call
            key_metrics=key_metrics,
            recent_news_sentiment=sentiment,
            business_highlights=highlights,
            risks=risks,
            financial_summary=result.financial_summary,
            generated_at=datetime.now().isoformat(),
        )

        self.logger.info(
            f"Company profile generation completed for {ticker}",
            company=company_name,
            sector=profile.sector,
            component="forward",
        )
        return profile


# Usage example and integration point for MCP
class CompanyProfileTool:
    """MCP tool wrapper for CompanyProfileGenerator"""

    def __init__(
        self,
        alphavantage_api_key: str = None,
        sentiment_tool=None,
        lm=None,
        config: Config = None,
        logger: LoggerManager = None,
    ):
        # Setup logging and config
        self.logger = logger or LoggerManager(
            name="company_profile_tool", level="DEBUG"
        )
        self.config = config or Config()

        self.logger.info("Initializing CompanyProfileTool", component="init")

        # Setup DSPy LM if not provided
        if lm:
            dspy.settings.configure(lm=lm)
        elif self.config:
            try:
                self.config.setup_dspy_lm()
                self.logger.info(
                    "DSPy LM configured from config",
                    model=self.config.model_name,
                    component="init",
                )
            except Exception as e:
                self.logger.error(
                    "Failed to setup DSPy LM from config",
                    error=str(e),
                    component="init",
                )

        self.generator = CompanyProfileGenerator(
            alphavantage_api_key=alphavantage_api_key,
            sentiment_tool=sentiment_tool,
            config=self.config,
            logger=self.logger,
        )

    async def generate_profile(self, company: str) -> dict:
        """Generate company profile for MCP"""
        try:
            self.logger.info(
                f"Generating profile for {company}", component="generate_profile"
            )
            profile = self.generator(company)

            result = {
                "success": True,
                "data": {
                    "company": profile.name,
                    "ticker": profile.ticker,
                    "description": profile.description,
                    "financial_summary": profile.financial_summary,
                    "market_cap": profile.market_cap,
                    "pe_ratio": profile.pe_ratio,
                    "dividend_yield": (
                        f"{profile.dividend_yield * 100:.2f}%"
                        if profile.dividend_yield
                        else None
                    ),
                    "sector": profile.sector,
                    "key_highlights": profile.business_highlights,
                    "key_risks": profile.risks,
                    "sentiment": profile.recent_news_sentiment,
                    "generated_at": profile.generated_at,
                },
            }

            self.logger.info(
                f"Successfully generated profile for {company}",
                component="generate_profile",
            )
            return result

        except Exception as e:
            self.logger.error(
                f"Failed to generate company profile for {company}",
                error=str(e),
                exc_info=True,
                component="generate_profile",
            )
            return {
                "success": False,
                "error": f"Failed to generate company profile: {str(e)}",
            }


# Example usage:
if __name__ == "__main__":
    # Setup logging and config
    logger = LoggerManager(name="company_profile_main", level="DEBUG")
    config = Config()

    logger.info("Starting Company Profile Generator", component="main")

    # You can test different LLM providers by uncommenting these:
    config.set_openrouter_config(
        "openrouter/deepseek/deepseek-chat-v3-0324:free"
    )  # OpenRouter - working well
    # config.set_ollama_config("ollama/qwen2.5-coder:3b")  # Default Ollama
    # config.set_nvidia_nim_config("nvidia/usdcode-llama-3.1-70b-instruct")  # Nvidia NIM

    try:
        logger.info("Initializing CompanyProfileGenerator", component="main")

        # Setup DSPy LM first
        config.setup_dspy_lm()
        logger.info("DSPy LM configured", model=config.model_name, component="main")

        # Create generator with config and logging
        generator = CompanyProfileGenerator(
            config=config,
            logger=logger,
            sentiment_tool=None,  # Replace with your sentiment tool if available
        )

        # Test with a company
        test_ticker = "AAPL"
        logger.info(f"Generating company profile for {test_ticker}", component="main")

        print("=" * 80)
        print("COMPANY PROFILE GENERATOR")
        print("=" * 80)

        profile = generator.forward(test_ticker)

        logger.info("Company profile generation completed", component="main")

        # Display results
        print(f"\n📊 COMPANY: {profile.name} ({profile.ticker})")
        print(f"🏢 SECTOR: {profile.sector} | INDUSTRY: {profile.industry}")
        if profile.market_cap:
            print(f"💰 MARKET CAP: {profile.market_cap}")
        if profile.pe_ratio:
            print(f"📈 P/E RATIO: {profile.pe_ratio:.2f}")

        print(f"\n📝 DESCRIPTION:")
        print(profile.description)

        print(f"\n✅ BUSINESS HIGHLIGHTS:")
        for highlight in profile.business_highlights:
            print(f"  • {highlight}")

        print(f"\n⚠️ KEY RISKS:")
        for risk in profile.risks:
            print(f"  • {risk}")

        print(f"\n💼 FINANCIAL SUMMARY:")
        print(profile.financial_summary)

        print(f"\n🕒 Generated at: {profile.generated_at}")

        # Test the MCP tool wrapper
        print("\n" + "=" * 80)
        print("TESTING MCP TOOL WRAPPER")
        print("=" * 80)

        tool = CompanyProfileTool(config=config, logger=logger)

        import asyncio

        async def test_tool():
            result = await tool.generate_profile("MSFT")
            if result["success"]:
                print(
                    f"✅ Successfully generated profile for {result['data']['company']}"
                )
                print(f"📊 Sector: {result['data']['sector']}")
                print(f"💰 Market Cap: {result['data']['market_cap']}")
            else:
                print(f"❌ Error: {result['error']}")

        asyncio.run(test_tool())

    except Exception as e:
        logger.error(
            "Error running company profile generator",
            error=str(e),
            exc_info=True,
            component="main",
        )
        print(f"\n❌ Error running company profile generator: {e}")
        print("\nTroubleshooting:")
        print("1. Check if API keys are set in environment variables")
        print("2. Verify LLM provider is accessible")
        print("3. Try different LLM provider by uncommenting config lines above")
        print("4. Check logs directory for detailed error information")
