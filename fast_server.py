import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import dspy
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from asset_price_tool import AssetPriceAnalyzer
from company_profile import CompanyProfileGenerator
from config import Config
from fundamental_tool import FundamentalAnalyzer
from macro_tool import MacroAnalyzer
from regulatory import RegulatoryAnalyzer
from sentiment_tool import NewsSentimentAnalyzer

load_dotenv()
chatbot_key = os.getenv("CHATBOT_KEY")


config = Config(
    model_name="openrouter/deepseek/deepseek-chat-v3-0324:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=chatbot_key,
    llm_temperature=0.0,
)

# Set up the language model
lm = config.setup_dspy_lm()
dspy.settings.configure(lm=lm)


app = FastAPI(title="MCP Financial Analysis Server")


# MCP Protocol Models
class MCPTool(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]


class MCPListToolsResponse(BaseModel):
    tools: List[MCPTool]


class MCPCallToolRequest(BaseModel):
    name: str
    arguments: Dict[str, Any]


class MCPCallToolResponse(BaseModel):
    content: List[Dict[str, Any]]
    isError: bool = False


class MCPErrorResponse(BaseModel):
    error: Dict[str, Any]
    isError: bool = True


# Your existing models
class ToolDiscoveryRequest(BaseModel):
    query: str
    execution_mode: Optional[str] = None


class ToolExecutionRequest(BaseModel):
    tools: List[str]
    parameters: Dict[str, Any]
    execution_mode: str = "standard"


class QueryAnalysisResponse(BaseModel):
    suggested_tools: List[str]
    execution_mode: str
    reasoning: str
    parameters: Dict[str, Any]


class MCPToolRegistry:
    def __init__(self):
        self.tools = {
            "analyze_price": {
                "description": "Analyze historical price data for a stock/crypto symbol",
                "parameters": ["symbol", "start_date", "end_date"],
                "categories": ["technical", "price", "market"],
                "execution_time": "fast",
                "dependencies": [],
                "mcp_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT)",
                        },
                        "start_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Start date for analysis (YYYY-MM-DD)",
                        },
                        "end_date": {
                            "type": "string",
                            "format": "date",
                            "description": "End date for analysis (YYYY-MM-DD)",
                        },
                    },
                    "required": ["symbol"],
                },
            },
            "analyze_sentiment": {
                "description": "Analyze news sentiment and market sentiment indicators for a stock",
                "parameters": ["symbol", "start_date", "end_date", "sources"],
                "categories": ["sentiment", "news", "market"],
                "execution_time": "medium",
                "dependencies": [],
                "mcp_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol to analyze sentiment for",
                        },
                        "start_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Start date for sentiment analysis",
                        },
                        "end_date": {
                            "type": "string",
                            "format": "date",
                            "description": "End date for sentiment analysis",
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "News sources to analyze",
                            "default": ["reuters", "bloomberg", "wsj"],
                        },
                    },
                    "required": ["symbol"],
                },
            },
            "analyze_fundamentals": {
                "description": "Analyze company fundamentals and financial statements",
                "parameters": [
                    "symbol",
                    "include_balance_sheet",
                    "include_cash_flow",
                    "include_earnings",
                    "quarters",
                ],
                "categories": ["fundamental", "financial", "earnings"],
                "execution_time": "medium",
                "dependencies": [],
                "mcp_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol for fundamental analysis",
                        },
                        "include_balance_sheet": {
                            "type": "boolean",
                            "description": "Include balance sheet analysis",
                            "default": True,
                        },
                        "include_cash_flow": {
                            "type": "boolean",
                            "description": "Include cash flow analysis",
                            "default": True,
                        },
                        "include_earnings": {
                            "type": "boolean",
                            "description": "Include earnings analysis",
                            "default": True,
                        },
                        "quarters": {
                            "type": "integer",
                            "description": "Number of quarters to analyze",
                            "default": 4,
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["symbol"],
                },
            },
            "analyze_macro": {
                "description": "Analyze macroeconomic factors and market conditions affecting a stock",
                "parameters": [
                    "symbol",
                    "include_sector_analysis",
                    "include_economic_indicators",
                ],
                "categories": ["macro", "economic", "sector"],
                "execution_time": "slow",
                "dependencies": [],
                "mcp_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol for macro analysis",
                        },
                        "include_sector_analysis": {
                            "type": "boolean",
                            "description": "Include sector-wide analysis",
                            "default": True,
                        },
                        "include_economic_indicators": {
                            "type": "boolean",
                            "description": "Include economic indicators analysis",
                            "default": True,
                        },
                    },
                    "required": ["symbol"],
                },
            },
            "generate_company_profile": {
                "description": "Generate comprehensive company profile and overview",
                "parameters": ["symbol"],
                "categories": ["profile", "company", "overview"],
                "execution_time": "fast",
                "dependencies": [],
                "mcp_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol for company profile",
                        }
                    },
                    "required": ["symbol"],
                },
            },
            "analyze_regulatory_compliance": {
                "description": "Analyze regulatory frameworks and compliance requirements",
                "parameters": [
                    "symbol",
                    "include_ai_regulations",
                    "include_sector_regulations",
                ],
                "categories": ["regulatory", "compliance", "legal"],
                "execution_time": "slow",
                "dependencies": ["generate_company_profile"],
                "mcp_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol for regulatory analysis",
                        },
                        "include_ai_regulations": {
                            "type": "boolean",
                            "description": "Include AI-specific regulations",
                            "default": True,
                        },
                        "include_sector_regulations": {
                            "type": "boolean",
                            "description": "Include sector-specific regulations",
                            "default": True,
                        },
                    },
                    "required": ["symbol"],
                },
            },
        }

    def get_mcp_tools(self) -> List[MCPTool]:
        """Return tools in MCP format"""
        mcp_tools = []
        for tool_name, tool_info in self.tools.items():
            mcp_tool = MCPTool(
                name=tool_name,
                description=tool_info["description"],
                inputSchema=tool_info["mcp_schema"],
            )
            mcp_tools.append(mcp_tool)
        return mcp_tools

    def get_all_tools(self) -> Dict[str, Any]:
        return self.tools

    def get_tools_by_category(self, categories: List[str]) -> List[str]:
        matching_tools = []
        for tool_name, tool_info in self.tools.items():
            if any(cat in tool_info["categories"] for cat in categories):
                matching_tools.append(tool_name)
        return matching_tools

    def get_sequential_dependencies(self, tool: str) -> List[str]:
        return self.tools.get(tool, {}).get("dependencies", [])


# Initialize registries
tool_registry = MCPToolRegistry()


# Your existing QueryAnalyzer (unchanged)
class QueryAnalyzer:
    def __init__(self, tool_registry: MCPToolRegistry):
        self.tool_registry = tool_registry

        self.sequential_keywords = [
            "step by step",
            "sequential",
            "workflow",
            "process",
            "first then",
            "after that",
            "following",
            "build upon",
            "depends on",
            "cascade",
            "regulatory analysis",
            "compliance check",
            "due diligence",
        ]

        self.parallel_keywords = [
            "comprehensive",
            "complete analysis",
            "all aspects",
            "overview",
            "summary",
            "report",
            "dashboard",
            "snapshot",
            "quick analysis",
        ]

        self.category_keywords = {
            "price": ["price", "chart", "technical", "trading", "movement", "trend"],
            "sentiment": ["sentiment", "news", "opinion", "market mood", "feeling"],
            "fundamental": [
                "fundamentals",
                "financial statements",
                "earnings",
                "revenue",
                "balance sheet",
            ],
            "macro": ["macro", "economic", "market conditions", "sector", "economy"],
            "profile": ["company", "profile", "overview", "background", "about"],
            "regulatory": ["regulatory", "compliance", "legal", "rules", "regulations"],
        }

    def analyze_query(
        self, query: str, execution_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        query_lower = query.lower()

        if not execution_mode:
            execution_mode = self._detect_execution_mode(query_lower)

        symbol = self._extract_symbol(query)
        relevant_categories = self._identify_categories(query_lower)
        suggested_tools = self._suggest_tools(relevant_categories, execution_mode)
        parameters = self._generate_parameters(query, symbol)

        return {
            "suggested_tools": suggested_tools,
            "execution_mode": execution_mode,
            "reasoning": self._generate_reasoning(
                query_lower, relevant_categories, execution_mode
            ),
            "parameters": parameters,
        }

    def _detect_execution_mode(self, query: str) -> str:
        sequential_score = sum(
            1 for keyword in self.sequential_keywords if keyword in query
        )
        parallel_score = sum(
            1 for keyword in self.parallel_keywords if keyword in query
        )

        if any(
            phrase in query
            for phrase in ["regulatory analysis", "compliance", "due diligence"]
        ):
            return "sequential"

        if sequential_score > parallel_score:
            return "sequential"
        else:
            return "standard"

    def _identify_categories(self, query: str) -> List[str]:
        relevant_categories = []
        for category, keywords in self.category_keywords.items():
            if any(keyword in query for keyword in keywords):
                relevant_categories.append(category)

        if not relevant_categories:
            relevant_categories = list(self.category_keywords.keys())

        return relevant_categories

    def _suggest_tools(self, categories: List[str], execution_mode: str) -> List[str]:
        tools = self.tool_registry.get_tools_by_category(categories)

        if execution_mode == "sequential":
            return self._order_tools_by_dependencies(tools)
        else:
            return tools

    def _order_tools_by_dependencies(self, tools: List[str]) -> List[str]:
        ordered = []
        remaining = tools.copy()

        while remaining:
            ready_tools = []
            for tool in remaining:
                deps = self.tool_registry.get_sequential_dependencies(tool)
                if all(dep in ordered for dep in deps):
                    ready_tools.append(tool)

            if not ready_tools:
                ready_tools = [remaining[0]]

            for tool in ready_tools:
                ordered.append(tool)
                remaining.remove(tool)

        return ordered

    def _extract_symbol(self, query: str) -> Optional[str]:
        import re

        symbols = re.findall(r"\b[A-Z]{1,5}\b", query)
        return symbols[0] if symbols else None

    def _generate_parameters(self, query: str, symbol: Optional[str]) -> Dict[str, Any]:
        params = {}
        if symbol:
            params["symbol"] = symbol

        import re

        dates = re.findall(r"\d{4}-\d{2}-\d{2}", query)
        if len(dates) >= 2:
            params["start_date"] = dates[0]
            params["end_date"] = dates[1]
        elif len(dates) == 1:
            params["end_date"] = dates[0]

        return params

    def _generate_reasoning(
        self, query: str, categories: List[str], execution_mode: str
    ) -> str:
        reasoning_parts = []
        reasoning_parts.append(f"Detected categories: {', '.join(categories)}")
        reasoning_parts.append(f"Execution mode: {execution_mode}")

        if execution_mode == "sequential":
            reasoning_parts.append(
                "Sequential mode chosen due to dependencies or step-by-step nature of query"
            )
        else:
            reasoning_parts.append(
                "Parallel mode chosen for efficient comprehensive analysis"
            )

        return " | ".join(reasoning_parts)


query_analyzer = QueryAnalyzer(tool_registry)


# MCP Tools


# Asset price analysis tool
async def analyze_price(symbol: str, start_date: str, end_date: str) -> dict:
    try:
        analyzer = AssetPriceAnalyzer(config=config)
        result = analyzer.forward(symbol, start_date=start_date, end_date=end_date)
        return {
            "tool": "analyze_price",
            "status": "success",
            "data": {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "price_trends": result.get("price_trends", "No trend data available"),
                "key_levels": result.get("key_levels", "No key levels available"),
                "analysis": f"Price analysis for {symbol} from {start_date} to {end_date} completed successfully",
            },
        }
    except Exception as e:
        return {
            "tool": "analyze_price",
            "status": "error",
            "data": {
                "symbol": symbol,
                "error": str(e),
                "analysis": f"Failed to analyze price for {symbol}: {str(e)}",
            },
        }


# Company profile tool/generator
async def generate_company_profile(symbol: str) -> dict:
    """Creates a company profile for a company´s ticker."""
    try:
        analyzer = CompanyProfileGenerator(config=config)
        summary = analyzer.forward(company_identifier=symbol)

        # Transform the result into the expected server format
        return {
            "tool": "generate_company_profile",
            "status": "success",
            "data": {
                "symbol": symbol,
                "company_name": getattr(summary, "name", "N/A"),
                "description": getattr(
                    summary, "description", "No description available"
                ),
                "sector": getattr(summary, "sector", "N/A"),
                "industry": getattr(summary, "industry", "N/A"),
                "market_cap": getattr(summary, "market_cap", "N/A"),
                "pe_ratio": getattr(summary, "pe_ratio", "N/A"),
                "business_highlights": getattr(summary, "business_highlights", []),
                "risks": getattr(summary, "risks", []),
                "financial_summary": getattr(
                    summary, "financial_summary", "No financial summary available"
                ),
                "analysis": f"Company profile for {symbol} generated successfully",
            },
        }

    except Exception as e:
        # Return error in the same standardized format
        return {
            "tool": "generate_company_profile",
            "status": "error",
            "data": {
                "symbol": symbol,
                "error": str(e),
                "analysis": f"Failed to generate company profile for {symbol}: {str(e)}",
            },
        }


# Fundamental data analyzer
async def analyze_fundamentals(symbol: str) -> dict:
    """Analyze company fundamentals including financial statements and earnings."""
    try:
        analyzer = FundamentalAnalyzer(config=config)
        result = analyzer.forward(
            symbol,
            include_balance_sheet=True,
            include_cash_flow=True,
            include_earnings=True,
        )

        # Transform the result into the expected server format
        return {
            "tool": "analyze_fundamentals",
            "status": "success",
            "data": {
                "symbol": symbol,
                "analysis": result.get("analysis", "No analysis available"),
                "data_sources": result.get("data_sources", []),
                "quarters_available": result.get("quarters_available", 0),
                "years_available": result.get("years_available", 0),
                "earnings_quarters_available": result.get(
                    "earnings_quarters_available", 0
                ),
                "earnings_years_available": result.get("earnings_years_available", 0),
                "formatted_data": result.get(
                    "formatted_data", "No formatted data available"
                ),
                "status_detail": result.get("status", "unknown"),
            },
        }

    except Exception as e:
        # Return error in the same standardized format
        return {
            "tool": "analyze_fundamentals",
            "status": "error",
            "data": {
                "symbol": symbol,
                "error": str(e),
                "analysis": f"Failed to analyze fundamentals for {symbol}: {str(e)}",
            },
        }


# Sentiment aanlysis tool
async def analyze_sentiment(symbol: str, start_date: str, end_date: str) -> dict:
    """Analyze news sentiment for a given symbol and date range."""
    try:
        analyzer = NewsSentimentAnalyzer(config=config)
        result = analyzer.forward(symbol, start_date=start_date, end_date=end_date)

        # Transform the result into the expected server format
        return {
            "tool": "analyze_sentiment",
            "status": "success",
            "data": {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "sentiment_label": result.get("sentiment_label", "Neutral"),
                "sentiment_score": result.get("sentiment_score", 0.5),
                "key_themes": result.get("key_themes", "Not specified"),
                "positive_indicators": result.get(
                    "positive_indicators", "Not specified"
                ),
                "negative_indicators": result.get(
                    "negative_indicators", "Not specified"
                ),
                "reasoning": result.get("reasoning", "No reasoning available"),
                "confidence": result.get("confidence", 0.0),
                "article_count": result.get("article_count", 0),
                "content_length": result.get("content_length", 0),
                "status_detail": result.get("status", "unknown"),
                "analysis": f"Sentiment analysis for {symbol} from {start_date} to {end_date} completed successfully",
            },
        }

    except Exception as e:
        # Return error in the same standardized format
        return {
            "tool": "analyze_sentiment",
            "status": "error",
            "data": {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "error": str(e),
                "analysis": f"Failed to analyze sentiment for {symbol}: {str(e)}",
            },
        }


# Macroeconomic analysis tool
async def analyze_macro(symbol: str) -> dict:
    """Analyze macroeconomic factors and their impact on the given symbol."""
    try:
        analyzer = MacroAnalyzer(config=config)
        result = analyzer.forward(symbol)

        # Transform the result into the expected server format
        return {
            "tool": "analyze_macro",
            "status": "success",
            "data": {
                "symbol": symbol,
                "impact": result.get("impact", "No impact analysis available"),
                "outlook": result.get("outlook", "No outlook available"),
                "data_series_count": result.get("data_series_count", 0),
                "sector": result.get("sector", "Unknown"),
                "data_points_per_series": result.get("data_points_per_series", 0),
                "status_detail": result.get("status", "unknown"),
                "analysis": f"Macroeconomic analysis for {symbol} completed successfully",
            },
        }

    except Exception as e:
        # Return error in the same standardized format
        return {
            "tool": "analyze_macro",
            "status": "error",
            "data": {
                "symbol": symbol,
                "error": str(e),
                "analysis": f"Failed to analyze macro factors for {symbol}: {str(e)}",
            },
        }


# Regulatory compliance tool (this is through RAG but shows the use of RAG inside MCP tools to do analysis takss from PDFs)
async def analyze_regulatory_compliance(
    symbol: str,
    include_ai_regulations: bool = True,
    include_sector_regulations: bool = True,
) -> dict:
    """Analyze regulatory compliance for a given symbol, based on AI regulatory data."""
    try:
        analyzer = RegulatoryAnalyzer(config=config)
        result = analyzer.analyze(ticker_symbol=symbol)

        # Extract key information from the regulatory analysis result
        # The result is a dict with questions as keys and analysis as values
        summary_findings = []
        summary_impacts = []
        summary_risks = []

        if isinstance(result, dict):
            for question, analysis in result.items():
                if isinstance(analysis, dict) and "error" not in analysis:
                    if "regulatory_findings" in analysis:
                        summary_findings.append(analysis["regulatory_findings"])
                    if "company_impact" in analysis:
                        summary_impacts.append(analysis["company_impact"])
                    if "risk_assessment" in analysis:
                        summary_risks.append(analysis["risk_assessment"])

        # Transform the result into the expected server format
        return {
            "tool": "analyze_regulatory_compliance",
            "status": "success",
            "data": {
                "symbol": symbol,
                "include_ai_regulations": include_ai_regulations,
                "include_sector_regulations": include_sector_regulations,
                "regulatory_findings": summary_findings,
                "company_impacts": summary_impacts,
                "risk_assessments": summary_risks,
                "full_analysis": result,
                "questions_analyzed": len(result) if isinstance(result, dict) else 0,
                "analysis": f"Regulatory compliance analysis for {symbol} completed successfully",
            },
        }

    except Exception as e:
        # Return error in the same standardized format
        return {
            "tool": "analyze_regulatory_compliance",
            "status": "error",
            "data": {
                "symbol": symbol,
                "include_ai_regulations": include_ai_regulations,
                "include_sector_regulations": include_sector_regulations,
                "error": str(e),
                "analysis": f"Failed to analyze regulatory compliance for {symbol}: {str(e)}",
            },
        }


# MCP Protocol Endpoints
@app.get("/mcp/tools", response_model=MCPListToolsResponse)
async def mcp_list_tools():
    """MCP-compliant tools listing endpoint"""
    tools = tool_registry.get_mcp_tools()
    return MCPListToolsResponse(tools=tools)


@app.post("/mcp/call")
async def mcp_call_tool(
    request: MCPCallToolRequest,
) -> Union[MCPCallToolResponse, MCPErrorResponse]:
    """MCP-compliant tool execution endpoint"""
    try:
        tool_name = request.name
        arguments = request.arguments

        # Execute the requested tool
        if tool_name == "analyze_price":
            result = await analyze_price(**arguments)
        elif tool_name == "analyze_sentiment":
            result = await analyze_sentiment(**arguments)
        elif tool_name == "analyze_fundamentals":
            result = await analyze_fundamentals(**arguments)
        elif tool_name == "analyze_macro":
            result = await analyze_macro(**arguments)
        elif tool_name == "generate_company_profile":
            result = await generate_company_profile(**arguments)
        elif tool_name == "analyze_regulatory_compliance":
            result = await analyze_regulatory_compliance(**arguments)
        else:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

        # Format response in MCP format
        content = [{"type": "text", "text": json.dumps(result, indent=2)}]

        return MCPCallToolResponse(content=content, isError=False)

    except Exception as e:
        error_content = [
            {"type": "text", "text": f"Error executing tool '{request.name}': {str(e)}"}
        ]
        return MCPCallToolResponse(content=error_content, isError=True)


# Your existing endpoints (unchanged)
@app.get("/")
async def root():
    return {"message": "MCP Financial Analysis Tools Server"}


@app.get("/tools/discover")
async def discover_tools() -> Dict[str, Any]:
    return {
        "tools": tool_registry.get_all_tools(),
        "total_count": len(tool_registry.get_all_tools()),
        "server_info": {
            "name": "MCP Financial Analysis Server",
            "version": "2.0.0",
            "capabilities": [
                "parallel_execution",
                "sequential_execution",
                "auto_discovery",
                "mcp_protocol",
            ],
        },
    }


@app.post("/query/analyze")
async def analyze_query(request: ToolDiscoveryRequest) -> QueryAnalysisResponse:
    analysis = query_analyzer.analyze_query(request.query, request.execution_mode)
    return QueryAnalysisResponse(**analysis)


@app.post("/tools/execute")
async def execute_tools(request: ToolExecutionRequest) -> Dict[str, Any]:
    if request.execution_mode == "sequential":
        return await _execute_sequential(request.tools, request.parameters)
    else:
        return await _execute_parallel(request.tools, request.parameters)


async def _execute_parallel(
    tools: List[str], parameters: Dict[str, Any]
) -> Dict[str, Any]:
    tasks = []
    for tool_name in tools:
        if tool_name == "analyze_price":
            task = analyze_price(**parameters)
        elif tool_name == "analyze_sentiment":
            task = analyze_sentiment(**parameters)
        elif tool_name == "analyze_fundamentals":
            task = analyze_fundamentals(**parameters)
        elif tool_name == "analyze_macro":
            task = analyze_macro(**parameters)
        elif tool_name == "generate_company_profile":
            task = generate_company_profile(**parameters)
        elif tool_name == "analyze_regulatory_compliance":
            task = analyze_regulatory_compliance(**parameters)

        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        "execution_mode": "parallel",
        "results": dict(zip(tools, results)),
        "timestamp": datetime.now().isoformat(),
    }


async def _execute_sequential(
    tools: List[str], parameters: Dict[str, Any]
) -> Dict[str, Any]:
    results = {}
    context = parameters.copy()

    for tool_name in tools:
        if tool_name == "analyze_price":
            result = await analyze_price(**context)
        elif tool_name == "analyze_sentiment":
            result = await analyze_sentiment(**context)
        elif tool_name == "analyze_fundamentals":
            result = await analyze_fundamentals(**context)
        elif tool_name == "analyze_macro":
            result = await analyze_macro(**context)
        elif tool_name == "generate_company_profile":
            result = await generate_company_profile(**context)
        elif tool_name == "analyze_regulatory_compliance":
            result = await analyze_regulatory_compliance(**context)

        results[tool_name] = result

        if isinstance(result, dict) and result.get("status") == "success":
            context.update(result.get("data", {}))

    return {
        "execution_mode": "sequential",
        "results": results,
        "final_context": context,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "available_tools": len(tool_registry.get_all_tools()),
        "mcp_compliant": True,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
