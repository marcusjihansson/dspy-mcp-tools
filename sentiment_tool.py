import json
from typing import Any, Dict, List, Optional, Union

import dspy
import requests
from pydantic import PrivateAttr

from base_tool import BaseFinancialTool
from config import Config
from log import LoggerManager
from utility import cache_result, retry_with_backoff


class NewsSentiment(dspy.Signature):
    """Analyze news sentiment for a ticker with detailed reasoning."""

    symbol = dspy.InputField(desc="Stock symbol")
    articles = dspy.InputField(desc="News articles content")
    sentiment_label = dspy.OutputField(
        desc="Sentiment category: Bullish, Bearish, or Neutral"
    )
    sentiment_score = dspy.OutputField(
        desc="Numerical sentiment score from 0.0 (very bearish) to 1.0 (very bullish), where 0.5 is neutral"
    )
    key_themes = dspy.OutputField(desc="Main themes and topics identified in the news")
    positive_indicators = dspy.OutputField(
        desc="Specific positive news points that support bullish sentiment"
    )
    negative_indicators = dspy.OutputField(
        desc="Specific negative news points that support bearish sentiment"
    )
    reasoning = dspy.OutputField(
        desc="Detailed explanation of how the sentiment score was determined"
    )
    confidence = dspy.OutputField(desc="Confidence level in the analysis (0-1)")


class NewsSentimentAnalyzer(BaseFinancialTool):
    """Analyze news sentiment using Finlight API v2 with requests library."""

    name: str = "NewsSentimentAnalyzer"
    input_type: type = str
    desc: str = (
        "Analyzes news sentiment for financial instruments using Finlight API v2"
    )
    tool: Optional[str] = None

    _analyze: Any = PrivateAttr()
    _base_url: str = PrivateAttr()
    _finlight_key: str = PrivateAttr()
    _session: requests.Session = PrivateAttr()

    def __init__(
        self, config: Optional[Config] = None, logger: Optional[LoggerManager] = None
    ):
        super().__init__(config, logger)
        self._analyze = dspy.ChainOfThought(NewsSentiment)
        self._base_url = "https://api.finlight.me/v2"

        if not self._config.finlight_key:
            raise ValueError("Finlight API key not found in config")

        self._finlight_key = self._config.finlight_key

        # Correct Finlight v2 auth style
        self._session = requests.Session()
        self._session.headers.update(
            {
                "X-API-KEY": self._finlight_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "NewsSentimentAnalyzer/1.0",
            }
        )

    def validate_and_parse_sentiment_score(self, score_str: str) -> float:
        try:
            score = float(score_str)
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            self._logger.warning(
                f"Invalid sentiment score '{score_str}', defaulting to 0.5"
            )
            return 0.5

    def interpret_sentiment_score(self, score: float) -> str:
        if score >= 0.7:
            return "Bullish"
        elif score <= 0.3:
            return "Bearish"
        else:
            return "Neutral"

    @retry_with_backoff(max_retries=3, retry_delay=1.0, logger=LoggerManager())
    @cache_result(cache_dir=".cache", logger=LoggerManager())
    def fetch_news(
        self, symbol: str, start_date: str, end_date: str
    ) -> List[Union[Dict, str]]:
        """Fetch extended news articles using Finlight API v2."""
        try:
            self._logger.info(
                f"Fetching extended news data for {symbol} from {start_date} to {end_date}"
            )

            endpoint = f"{self._base_url}/articles"
            payload = {
                "query": symbol,
                "from": start_date,
                "to": end_date,
                "limit": 10,
                "language": "en",
                "includeContent": True,
                "includeEntities": True,
            }

            self._logger.debug(f"POST {endpoint}")
            self._logger.debug(f"Payload: {payload}")

            response = self._session.post(endpoint, json=payload, timeout=30)
            self._logger.debug(f"Status: {response.status_code}")
            self._logger.debug(f"Headers: {dict(response.headers)}")

            response.raise_for_status()

            try:
                data = response.json()
            except json.JSONDecodeError as e:
                self._logger.error(f"Failed to parse JSON: {e}")
                self._logger.debug(f"Raw text: {response.text[:500]}...")
                raise ValueError(f"Invalid JSON response from API: {e}")

            articles = []
            if isinstance(data, dict):
                if "articles" in data and isinstance(data["articles"], list):
                    articles = data["articles"]
                elif "data" in data and isinstance(data["data"], list):
                    articles = data["data"]
                else:
                    self._logger.warning(
                        f"No recognizable articles key. Keys: {list(data.keys())}"
                    )
            elif isinstance(data, list):
                articles = data
            else:
                raise ValueError(f"Unexpected response format: {type(data)}")

            if not isinstance(articles, list):
                raise ValueError(f"Invalid articles format: {type(articles)}")

            self._logger.info(f"Retrieved {len(articles)} articles for {symbol}")
            return articles

        except requests.exceptions.Timeout:
            self._logger.error(f"Timeout fetching articles for {symbol}")
            raise
        except requests.exceptions.ConnectionError:
            self._logger.error(f"Connection error fetching articles for {symbol}")
            raise
        except requests.exceptions.HTTPError as e:
            self._logger.error(
                f"HTTP {e.response.status_code} fetching articles for {symbol}: {e}"
            )
            try:
                err = e.response.json()
                if "error" in err:
                    raise ValueError(f"API error: {err['error']}")
                elif "message" in err:
                    raise ValueError(f"API error: {err['message']}")
            except (json.JSONDecodeError, AttributeError):
                pass
            raise
        except requests.exceptions.RequestException as e:
            self._logger.error(f"Request error fetching articles for {symbol}: {e}")
            raise
        except Exception as e:
            self._logger.error(f"Error fetching articles for {symbol}: {e}")
            raise

    def process_articles(self, articles: List[Union[Dict, str]]) -> str:
        if not articles:
            return ""
        content_parts = []
        for i, article in enumerate(articles):
            try:
                self._logger.debug(f"Processing article {i}, type: {type(article)}")
                article_text = ""
                if isinstance(article, dict):
                    title = article.get("title", article.get("headline", ""))
                    content = article.get(
                        "content",
                        article.get(
                            "body", article.get("text", article.get("description", ""))
                        ),
                    )
                    summary = article.get(
                        "summary", article.get("snippet", article.get("excerpt", ""))
                    )
                    text_parts = [title, summary, content]
                    article_text = " ".join(
                        filter(None, [part.strip() for part in text_parts if part])
                    )
                elif isinstance(article, str):
                    article_text = article.strip()
                else:
                    self._logger.warning(f"Unsupported article format: {type(article)}")
                    continue
                if article_text:
                    content_parts.append(article_text)
                    self._logger.debug(
                        f"Processed article {i}: {len(article_text)} chars"
                    )
            except Exception as e:
                self._logger.warning(f"Error processing article {i}: {e}")
                continue
        combined_content = " ".join(content_parts)
        self._logger.info(
            f"Processed {len(articles)} articles into {len(combined_content)} chars of text"
        )
        return combined_content

    def forward(self, symbol: str, **kwargs) -> Dict[str, Any]:
        if not self.validate_inputs(symbol):
            return {
                "symbol": symbol,
                "error": f"Invalid symbol provided: {symbol}",
                "sentiment_label": "Analysis unavailable due to invalid input",
                "sentiment_score": 0.5,
                "reasoning": "Input validation failed",
                "confidence": 0.0,
                "status": "invalid_input",
            }
        try:
            start_date = kwargs.get("start_date")
            end_date = kwargs.get("end_date")
            if not start_date or not end_date:
                return {
                    "symbol": symbol,
                    "error": "start_date and end_date are required",
                    "sentiment_label": "Analysis unavailable due to missing dates",
                    "sentiment_score": 0.5,
                    "reasoning": "Required date parameters missing",
                    "confidence": 0.0,
                    "status": "missing_dates",
                }
            articles = self.fetch_news(symbol, start_date, end_date)
            if not articles:
                return {
                    "symbol": symbol,
                    "sentiment_label": "Neutral",
                    "sentiment_score": 0.5,
                    "reasoning": "No recent news articles found",
                    "confidence": 0.0,
                    "article_count": 0,
                    "date_range": f"{start_date} to {end_date}",
                    "status": "no_news",
                }
            content = self.process_articles(articles)
            if not content.strip():
                return {
                    "symbol": symbol,
                    "sentiment_label": "Neutral",
                    "sentiment_score": 0.5,
                    "reasoning": "Articles found but no analyzable text content",
                    "confidence": 0.0,
                    "article_count": len(articles),
                    "date_range": f"{start_date} to {end_date}",
                    "status": "no_content",
                }
            self._logger.debug(
                f"Content for DSPy analysis (len={len(content)}): {content[:500]}..."
            )
            analysis = self._analyze(symbol=symbol, articles=content)
            required_attrs = [
                "sentiment_label",
                "sentiment_score",
                "reasoning",
                "confidence",
            ]
            for attr in required_attrs:
                if not hasattr(analysis, attr):
                    raise ValueError(
                        f"Missing required attribute '{attr}' in DSPy analysis output"
                    )
            sentiment_score = self.validate_and_parse_sentiment_score(
                analysis.sentiment_score
            )
            sentiment_label = (
                analysis.sentiment_label
                or self.interpret_sentiment_score(sentiment_score)
            )
            return {
                "symbol": symbol,
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
                "key_themes": getattr(analysis, "key_themes", "Not specified"),
                "positive_indicators": getattr(
                    analysis, "positive_indicators", "Not specified"
                ),
                "negative_indicators": getattr(
                    analysis, "negative_indicators", "Not specified"
                ),
                "reasoning": analysis.reasoning,
                "confidence": float(analysis.confidence),
                "article_count": len(articles),
                "date_range": f"{start_date} to {end_date}",
                "content_length": len(content),
                "status": "success",
            }
        except Exception as e:
            self._logger.error(f"Sentiment analysis error for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "sentiment_label": "Error",
                "sentiment_score": 0.5,
                "reasoning": f"Analysis failed due to error: {str(e)}",
                "confidence": 0.0,
                "status": "error",
            }

    def validate_inputs(self, symbol: str) -> bool:
        if not super().validate_inputs(symbol):
            return False
        if len(symbol) > 10:
            self._logger.warning(f"Symbol too long: {symbol}")
            return False
        return True

    def __del__(self):
        if hasattr(self, "_session"):
            self._session.close()


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    from config import Config

    load_dotenv()
    chatbot_key = os.getenv("CHATBOT_KEY")

    config_custom = Config(
        model_name="openrouter/deepseek/deepseek-chat-v3-0324:free",
        api_base="https://openrouter.ai/api/v1",
        api_key=chatbot_key,
        llm_temperature=0.0,
    )

    lm = config_custom.setup_dspy_lm()
    dspy.settings.configure(lm=lm)

    sentiment_analyzer = NewsSentimentAnalyzer(config_custom)

    result = sentiment_analyzer.forward(
        "AAPL", start_date="2025-07-30", end_date="2025-08-09"
    )

    print("Analysis Result:")
    print("________________________________________________")
    print(f"Symbol: {result['symbol']}")
    print("________________________________________________")
    print(f"Status: {result['status']}")
    print("________________________________________________")
    print(f"Sentiment Label: {result.get('sentiment_label', 'N/A')}")
    print("________________________________________________")
    print(f"Sentiment Score: {result.get('sentiment_score', 'N/A')}")
    print("________________________________________________")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print("________________________________________________")
    print(f"Key Themes: {result.get('key_themes', 'N/A')}")
    print("________________________________________________")
    print(f"Positive Indicators: {result.get('positive_indicators', 'N/A')}")
    print("________________________________________________")
    print(f"Negative Indicators: {result.get('negative_indicators', 'N/A')}")
    print("________________________________________________")
    print(f"Reasoning: {result.get('reasoning', 'N/A')}")
    print("________________________________________________")
