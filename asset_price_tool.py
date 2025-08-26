# Asset price tool.py
# Enhanced version with crypto support through Alpaca API

from datetime import datetime, timedelta
from typing import Any, Dict, Literal, Optional

import dspy
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from pydantic import PrivateAttr

from base_tool import BaseFinancialTool
from config import Config
from log import LoggerManager
from utility import cache_result, retry_with_backoff


class PriceAnalysis(dspy.Signature):
    """Analyze price trends and patterns."""

    symbol = dspy.InputField(desc="Stock or crypto symbol")
    asset_type = dspy.InputField(desc="Type of asset (stock or crypto)")
    timeframe = dspy.InputField(desc="Timeframe for analysis")
    price_data = dspy.InputField(desc="Summary of price data")
    price_trends = dspy.OutputField(desc="Analysis of price trends")
    key_levels = dspy.OutputField(desc="Key support/resistance levels")


class AssetPriceAnalyzer(BaseFinancialTool):
    """Analyze asset price data for both stocks and crypto using Alpaca API."""

    name: str = "AssetPriceAnalyzer"
    input_type: type = str
    desc: str = (
        "Analyzes stock and crypto price data and identifies trends and key levels"
    )
    tool: Optional[str] = None

    # Declare analyze as a private attribute for Pydantic compatibility
    _analyze: Any = PrivateAttr()
    _stock_client: Any = PrivateAttr()
    _crypto_client: Any = PrivateAttr()

    def __init__(
        self, config: Optional[Config] = None, logger: Optional[LoggerManager] = None
    ):
        super().__init__(config, logger)
        self._analyze = dspy.ChainOfThought(PriceAnalysis)

        # Get API credentials from config - use private attribute
        if not self._config.alpaca_key or not self._config.alpaca_secret:
            raise ValueError("Alpaca API credentials not found in config")

        # Initialize both stock and crypto clients
        self._stock_client = StockHistoricalDataClient(
            self._config.alpaca_key, self._config.alpaca_secret
        )

        self._crypto_client = CryptoHistoricalDataClient(
            self._config.alpaca_key, self._config.alpaca_secret
        )

    def detect_asset_type(self, symbol: str) -> Literal["stock", "crypto"]:
        """Detect if symbol is a stock or crypto based on common patterns."""
        # Common crypto symbols often have USD, USDT, or are well-known crypto pairs
        crypto_patterns = [
            "BTC",
            "ETH",
            "ADA",
            "DOT",
            "SOL",
            "AVAX",
            "MATIC",
            "LINK",
            "UNI",
            "AAVE",
            "LTC",
            "XRP",
            "DOGE",
            "SHIB",
            "BCH",
            "ETC",
            "FIL",
            "ATOM",
            "ALGO",
            "XTZ",
        ]

        # Check if symbol ends with USD (common for crypto pairs)
        if symbol.endswith("USD") or symbol.endswith("USDT"):
            return "crypto"

        # Check if base symbol (without USD) is a known crypto
        base_symbol = symbol.replace("USD", "").replace("USDT", "")
        if base_symbol in crypto_patterns:
            return "crypto"

        # Default to stock for traditional symbols
        return "stock"

    @retry_with_backoff(max_retries=3, retry_delay=1.0, logger=LoggerManager())
    @cache_result(cache_dir=".cache", logger=LoggerManager())
    def fetch_stock_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch stock OHLCV data from Alpaca."""
        self._logger.info(f"Fetching stock price data for {symbol}")
        try:
            now = datetime.now()
            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=now - timedelta(days=days),
                limit=days,
            )
            bars = self._stock_client.get_stock_bars(req)

            if bars.df.empty:
                self._logger.info(f"No stock price data returned for {symbol}")
                return pd.DataFrame()

            return bars.df
        except Exception as e:
            self._logger.error(f"Error fetching stock price for {symbol}: {e}")
            raise

    @retry_with_backoff(max_retries=3, retry_delay=1.0, logger=LoggerManager())
    @cache_result(cache_dir=".cache", logger=LoggerManager())
    def fetch_crypto_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch crypto OHLCV data from Alpaca."""
        self._logger.info(f"Fetching crypto price data for {symbol}")
        try:
            now = datetime.now()

            # Ensure symbol is in correct format for crypto (e.g., BTC/USD)
            if "/" not in symbol and not symbol.endswith("USD"):
                symbol = f"{symbol}/USD"

            req = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=now - timedelta(days=days),
                limit=days,
            )
            bars = self._crypto_client.get_crypto_bars(req)

            if bars.df.empty:
                self._logger.info(f"No crypto price data returned for {symbol}")
                return pd.DataFrame()

            return bars.df
        except Exception as e:
            self._logger.error(f"Error fetching crypto price for {symbol}: {e}")
            raise

    def fetch_price_data(
        self, symbol: str, days: int = 30, asset_type: Optional[str] = None
    ) -> tuple[pd.DataFrame, str]:
        """Fetch price data for either stock or crypto."""
        if asset_type is None:
            asset_type = self.detect_asset_type(symbol)

        if asset_type == "crypto":
            df = self.fetch_crypto_data(symbol, days)
        else:
            df = self.fetch_stock_data(symbol, days)

        return df, asset_type

    def summarize_price_data(self, df: pd.DataFrame, asset_type: str) -> str:
        """Create summary of price data for analysis."""
        if df.empty:
            return "No price data available"

        # Format currency based on asset type
        currency_symbol = "$" if asset_type == "stock" else "$"

        return f"""
        {asset_type.title()} Price Summary (last {len(df)} days):
        - Current: {currency_symbol}{df["close"].iloc[-1]:.2f}
        - High: {currency_symbol}{df["high"].max():.2f}
        - Low: {currency_symbol}{df["low"].min():.2f}
        - Avg Volume: {df["volume"].mean():,.0f}
        - Price Change: {((df["close"].iloc[-1] / df["close"].iloc[0]) - 1) * 100:.1f}%
        - Asset Type: {asset_type.title()}
        """

    def forward(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Process the price analysis request for stocks or crypto."""
        # Validate inputs using base class method
        if not self.validate_inputs(symbol):
            return {
                "error": f"Invalid symbol provided: {symbol}",
                "price_trends": "Analysis unavailable due to invalid input",
                "key_levels": "Analysis unavailable due to invalid input",
                "asset_type": "unknown",
            }

        try:
            # Extract parameters from kwargs
            start_date = kwargs.get("start_date")
            end_date = kwargs.get("end_date")
            asset_type = kwargs.get("asset_type")  # Allow manual override

            # Calculate days from date range if provided
            days = 30
            if start_date and end_date:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                days = (end - start).days

            # Fetch and analyze data
            df, detected_asset_type = self.fetch_price_data(symbol, days, asset_type)

            if df.empty:
                return {
                    "symbol": symbol,
                    "asset_type": detected_asset_type,
                    "timeframe": f"{days} days",
                    "price_trends": f"No price data available for {symbol}",
                    "key_levels": "Unable to determine key levels without data",
                    "status": "no_data",
                }

            # Generate summary and perform analysis
            summary = self.summarize_price_data(df, detected_asset_type)
            analysis = self._analyze(
                symbol=symbol,
                asset_type=detected_asset_type,
                timeframe=f"{days} days",
                price_data=summary,
            )

            return {
                "symbol": symbol,
                "asset_type": detected_asset_type,
                "timeframe": f"{days} days",
                "price_trends": analysis.price_trends,
                "key_levels": analysis.key_levels,
                "raw_data_summary": summary,
                "status": "success",
            }

        except Exception as e:
            self._logger.error(f"Price analysis error for {symbol}: {e}")
            return {
                "symbol": symbol,
                "asset_type": kwargs.get("asset_type", "unknown"),
                "error": str(e),
                "price_trends": f"Error analyzing {symbol}: {str(e)}",
                "key_levels": "Analysis unavailable due to error",
                "status": "error",
            }

    def validate_inputs(self, symbol: str) -> bool:
        """Enhanced input validation for both stock and crypto analysis."""
        # Use base class validation first
        if not super().validate_inputs(symbol):
            return False

        # Additional validation for both stocks and crypto
        if len(symbol) > 15:  # Allow longer symbols for crypto pairs like BTC/USD
            self._logger.warning(f"Symbol too long: {symbol}")
            return False

        # Allow alphanumeric and common crypto symbols with / and -
        if not all(c.isalnum() or c in ["/", "-", "."] for c in symbol):
            self._logger.warning(f"Invalid symbol format: {symbol}")
            return False

        return True


if __name__ == "__main__":
    # Initialize the tool
    import os

    from dotenv import load_dotenv

    from config import Config

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

    # Pass the config object, not the language model
    analyzer = AssetPriceAnalyzer(config_custom)

    # This analysis works with both stocks and crypto
    print("=== Price ANALYSIS ===")
    result = analyzer.forward(
        symbol="GOOG", start_date="2024-03-01", end_date="2025-08-01"
    )
    print("Price Trends:")
    print("________________________________________________")
    print(result["price_trends"])
    print("________________________________________________")
    print("\nStock - Key Levels:")
    print("________________________________________________")
    print(result["key_levels"])
    print("________________________________________________")
    print(result["status"])
    print("________________________________________________")
