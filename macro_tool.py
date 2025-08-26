# ============================================================================
# Macroeconomic Analysis
# ============================================================================

from typing import Any, Dict, Optional

import dspy
import pandas as pd
from dbnomics import fetch_series
from pydantic import PrivateAttr

from base_tool import BaseFinancialTool
from config import Config
from log import LoggerManager
from utility import cache_result, retry_with_backoff


class MacroImpact(dspy.Signature):
    """Analyze macroeconomic impact."""

    symbol = dspy.InputField(desc="Stock symbol")
    macro_data = dspy.InputField(desc="Macroeconomic data summary")
    sector = dspy.InputField(desc="Company sector information")
    impact = dspy.OutputField(desc="Impact analysis")
    outlook = dspy.OutputField(desc="Future outlook")


class MacroAnalyzer(BaseFinancialTool):
    """Analyze macroeconomic data from DBnomics."""

    name: str = "MacroAnalyzer"
    input_type: type = str
    desc: str = (
        "Analyzes macroeconomic impact on financial instruments using DBnomics data"
    )
    tool: Optional[str] = None

    # Declare analyze as a private attribute for Pydantic compatibility
    _analyze: Any = PrivateAttr()
    _series: Dict[str, str] = PrivateAttr()
    _sectors: Dict[str, str] = PrivateAttr()

    def __init__(
        self, config: Optional[Config] = None, logger: Optional[LoggerManager] = None
    ):
        super().__init__(config, logger)
        self._analyze = dspy.ChainOfThought(MacroImpact)

        # Macroeconomic data series from DBnomics
        self._series = {
            "BEA/NIPA-T10701/A191RL-Q": "GDP Quarterly",
            "BLS/cu/CUSR0000SA0": "Monthly US CPI",
            "FED/H15/RIFLGFCY10_N.B": "US 10-year Treasury",
            "FED/H15/RIFLGFCY02_N.B": "US 2-year Treasury",
            "INSEE/IPI-2021/A.BDM.IPI_MOYENNE_ANNUELLE.MIG_CAG.SO.MOYENNE_ANNUELLE.FM.SO.BRUT.2021": "Capital Industrial Index",
            "INSEE/IPI-2021/A.BDM.IPI_MOYENNE_ANNUELLE.SO.26-1.MOYENNE_ANNUELLE.FM.SO.BRUT.2021": "Semiconductor Index",
            "INSEE/IPI-2021/A.BDM.IPI_MOYENNE_ANNUELLE.SO.26-2.MOYENNE_ANNUELLE.FM.SO.BRUT.2021": "Compute Index",
            "INSEE/IPI-2021/A.BDM.IPI_MOYENNE_ANNUELLE.MIG_NRG.SO.MOYENNE_ANNUELLE.FM.SO.BRUT.2021": "Energy Index",
        }

        # Company sector profiles for targeted analysis
        self._sectors = {
            "AAPL": "Technology hardware manufacturer focusing on consumer electronics, software, and services",
            "MSFT": "Software and cloud computing company with diverse enterprise and consumer offerings",
            "GOOGL": "Technology company specializing in internet services, digital advertising, cloud computing, and AI",
            "META": "Social media and metaverse technology company with focus on digital advertising",
            "AMZN": "E-commerce, cloud computing, digital streaming, and AI company with diverse revenue streams",
            "TSLA": "Electric vehicle and clean energy company with significant manufacturing and AI components",
            "NVDA": "Semiconductor company specializing in graphics processing units and AI acceleration hardware",
            "AMD": "Semiconductor company producing processors for computing and graphics applications",
            "INTC": "Semiconductor manufacturer specializing in CPUs, data center, and other computing technologies",
            "IBM": "Technology and consulting company with focus on cloud computing, AI, and enterprise services",
        }

    @retry_with_backoff(max_retries=3, retry_delay=2.0, logger=LoggerManager())
    @cache_result(cache_dir=".cache", logger=LoggerManager())
    def fetch_macro_data(self, data_points: int = 5) -> Dict[str, pd.DataFrame]:
        """Fetch macroeconomic data from DBnomics."""
        self._logger.info(
            f"Fetching macroeconomic data ({data_points} recent points per series)"
        )

        try:
            data = {}
            successful_fetches = 0

            for series_code, series_name in self._series.items():
                try:
                    self._logger.debug(f"Fetching {series_name} ({series_code})")
                    df = fetch_series(series_code)

                    if df is not None and not df.empty:
                        # Get the most recent data points
                        recent_data = df[["original_period", "original_value"]].tail(
                            data_points
                        )
                        data[series_name] = recent_data
                        successful_fetches += 1
                        self._logger.debug(
                            f"Successfully fetched {len(recent_data)} points for {series_name}"
                        )
                    else:
                        self._logger.warning(f"No data returned for {series_name}")

                except Exception as series_error:
                    self._logger.warning(
                        f"Failed to fetch {series_name}: {series_error}"
                    )
                    continue

            self._logger.info(
                f"Successfully fetched {successful_fetches}/{len(self._series)} macro series"
            )

            if successful_fetches == 0:
                raise Exception("No macroeconomic data could be retrieved")

            return data

        except Exception as e:
            self._logger.error(f"Error fetching macroeconomic data: {e}")
            raise

    def format_macro_data(self, macro_data: Dict[str, pd.DataFrame]) -> str:
        """Format macroeconomic data for analysis."""
        if not macro_data:
            return "No macroeconomic data available"

        formatted_sections = []

        for series_name, df in macro_data.items():
            if df is not None and not df.empty:
                # Create a clean representation of the data
                data_summary = f"\n{series_name}:"
                data_summary += f"\n{df.to_string(index=False)}"

                # Add trend information if we have multiple points
                if len(df) > 1:
                    try:
                        first_val = pd.to_numeric(
                            df["original_value"].iloc[0], errors="coerce"
                        )
                        last_val = pd.to_numeric(
                            df["original_value"].iloc[-1], errors="coerce"
                        )

                        if (
                            pd.notna(first_val)
                            and pd.notna(last_val)
                            and first_val != 0
                        ):
                            change_pct = ((last_val - first_val) / first_val) * 100
                            trend = (
                                "↑"
                                if change_pct > 0
                                else "↓" if change_pct < 0 else "→"
                            )
                            data_summary += f"\nTrend: {trend} {change_pct:.1f}%"
                    except:
                        pass  # Skip trend calculation if data isn't numeric

                formatted_sections.append(data_summary)

        return "\n".join(formatted_sections)

    def get_sector_info(self, symbol: str) -> str:
        """Get sector information for the given symbol."""
        sector_info = self._sectors.get(symbol.upper())
        if sector_info:
            return sector_info
        else:
            self._logger.info(
                f"No sector information available for {symbol}, using generic analysis"
            )
            return "General financial instrument"

    def forward(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Process the macroeconomic analysis request."""
        # Validate inputs using base class method
        if not self.validate_inputs(symbol):
            return {
                "symbol": symbol,
                "error": f"Invalid symbol provided: {symbol}",
                "impact": "Analysis unavailable due to invalid input",
                "outlook": "Analysis unavailable due to invalid input",
                "status": "invalid_input",
            }

        try:
            # Extract optional parameters
            data_points = kwargs.get("data_points", 5)

            # Fetch macroeconomic data
            macro_data = self.fetch_macro_data(data_points)

            if not macro_data:
                return {
                    "symbol": symbol,
                    "impact": "No macroeconomic data available for analysis",
                    "outlook": "Unable to provide outlook without macro data",
                    "data_series_count": 0,
                    "status": "no_data",
                }

            # Format data and get sector information
            formatted_data = self.format_macro_data(macro_data)
            sector_info = self.get_sector_info(symbol)

            # Perform macroeconomic impact analysis
            analysis = self._analyze(
                symbol=symbol, macro_data=formatted_data, sector=sector_info
            )

            return {
                "symbol": symbol,
                "impact": analysis.impact,
                "outlook": analysis.outlook,
                "data_series_count": len(macro_data),
                "sector": sector_info,
                "data_points_per_series": data_points,
                "status": "success",
            }

        except Exception as e:
            self._logger.error(f"Macro analysis error for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "impact": f"Error analyzing macroeconomic impact: {str(e)}",
                "outlook": "Analysis unavailable due to error",
                "status": "error",
            }

    def validate_inputs(self, symbol: str) -> bool:
        """Enhanced input validation for macro analysis."""
        # Use base class validation first
        if not super().validate_inputs(symbol):
            return False

        # Additional validation specific to macro analysis
        if len(symbol) > 10:
            self._logger.warning(f"Symbol too long: {symbol}")
            return False

        return True

    def add_sector(self, symbol: str, description: str) -> None:
        """Add a new sector description for a symbol."""
        self._sectors[symbol.upper()] = description
        self._logger.info(f"Added sector information for {symbol}: {description}")

    def get_available_series(self) -> Dict[str, str]:
        """Get information about available macroeconomic data series."""
        return self._series.copy()


if __name__ == "__main__":
    # Initialize the tool
    import os

    from dotenv import load_dotenv

    from config import Config

    config = Config()

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
    macro_analyzer = MacroAnalyzer(config_custom)

    result = macro_analyzer.forward("AAPL")

    print("Analysis Result:")
    print("________________________________________________")
    print(f"Symbol: {result['symbol']}")
    print("________________________________________________")
    print(f"Status: {result['status']}")
    print("________________________________________________")
    print(f"Impact: {result['impact']}")
    print("________________________________________________")
    print(f"Outlook: {result['outlook']}")
    print("________________________________________________")
