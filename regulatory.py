import os
from typing import Any, Dict, List, Optional

import dspy
import numpy as np
import ollama
import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import AsIs, register_adapter

from config import Config

# Import our logging and config systems
from log import LoggerManager

# Load environment variables
load_dotenv()


# Register NumPy adapter for psycopg2
def adapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)


def adapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)


register_adapter(np.float64, adapt_numpy_float64)
register_adapter(np.float32, adapt_numpy_float32)


class PostgresRetriever(dspy.Retrieve):
    """Custom PostgreSQL retriever using the ai_law_db database."""

    def __init__(
        self,
        conn,
        model_name: str = "bge-m3:latest",
        k: int = 3,
    ):
        super().__init__(k=k)
        self.conn = conn
        self.model_name = model_name

    def forward(self, query: str, k: int = None) -> dspy.Prediction:
        k = k or self.k

        try:
            # Generate query embedding
            response = ollama.embeddings(model=self.model_name, prompt=query)
            query_embedding = response["embedding"]

            # Retrieve similar documents from database
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT content, metadata, (1 - (embedding <=> %s::vector)) as similarity
                FROM rag_documents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_embedding, query_embedding, k),
            )

            results = cur.fetchall()
            cur.close()

            # Extract just the content for passages
            passages = [result[0] for result in results]

            return dspy.Prediction(passages=passages)

        except Exception as e:
            print(f"Error in retrieval: {e}")
            return dspy.Prediction(passages=[])


class RegulatoryAnalyzer:
    def __init__(self, config: Config = None, logger: LoggerManager = None):
        """
        Initializes the RegulatoryAnalyzer using the PostgreSQL database.
        Uses a two-stage approach: first understanding regulations, then applying to specific companies.
        """
        # Setup logging and config
        self.logger = logger or LoggerManager(name="regulatory_analyzer", level="DEBUG")
        self.config = config or Config()

        self.logger.info(
            "Initializing RegulatoryAnalyzer with PostgreSQL database", component="init"
        )

        self.regulation_questions = [
            "What are the key requirements of recent AI regulations?",
            "What industries or business activities face the most regulatory scrutiny?",
            "What compliance obligations do AI regulations impose on companies?",
            """Which countries (China, United Kingdom, United States), regions (EU Council, EU) and or groups (G7, OECD) 
                have the most strict AI regulations that could be imposed on companies?""",
            """Which countries (for example; China, United Kingdom, United States), regions (for example; EU Council, EU) and or groups (for example; G7, OECD) 
                have the most lenient AI regulations that could be imposed on companies?""",
        ]
        self._setup_pipeline()

    def _setup_database_connection(self):
        """Setup database connection"""
        try:
            conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
            )
            self.logger.info("Database connection established", component="database")
            return conn
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}", component="database")
            raise

    def _setup_pipeline(self):
        """Sets up a two-stage DSPy pipeline with PostgreSQL retriever."""

        # Configure models using our config system with detailed logging
        try:
            self.logger.info("Setting up DSPy language model", component="setup")

            # Use config system to setup LM - try different providers if needed
            try:
                # First try with current config
                self.logger.debug(
                    f"Attempting to configure with model: {self.config.model_name}",
                    component="setup",
                )
                generation_lm = self.config.setup_dspy_lm()
                self.logger.info(
                    "Successfully configured DSPy with primary model",
                    model=self.config.model_name,
                    component="setup",
                )

            except Exception as e:
                self.logger.warning(
                    "Primary model failed, trying Ollama fallback",
                    error=str(e),
                    component="setup",
                )
                # Fallback to Ollama with available models
                self.config.set_ollama_config("ollama/qwen2.5-coder:3b")
                try:
                    generation_lm = self.config.setup_dspy_lm()
                    self.logger.info(
                        "Successfully configured DSPy with Ollama fallback",
                        model=self.config.model_name,
                        component="setup",
                    )
                except Exception as e2:
                    self.logger.warning(
                        "First Ollama fallback failed, trying second",
                        error=str(e2),
                        component="setup",
                    )
                    # Second fallback
                    self.config.set_ollama_config("ollama/qwen3:1.7b")
                    generation_lm = self.config.setup_dspy_lm()
                    self.logger.info(
                        "Successfully configured DSPy with second Ollama fallback",
                        model=self.config.model_name,
                        component="setup",
                    )

        except Exception as e:
            self.logger.error(
                "All DSPy configuration attempts failed",
                error=str(e),
                exc_info=True,
                component="setup",
            )
            raise

        # Setup database connection and retriever
        self.logger.info(
            "Setting up database connection and retriever", component="setup"
        )
        self.conn = self._setup_database_connection()

        # Create PostgreSQL retriever
        try:
            self.logger.info(
                "Creating PostgreSQL retriever with embeddings", component="retriever"
            )
            self.retriever = PostgresRetriever(
                self.conn,
                model_name="bge-m3:latest",  # Use the available embedding model
            )
            self.logger.info(
                "Successfully created PostgreSQL retriever",
                model="bge-m3:latest",
                component="retriever",
            )
        except Exception as e:
            self.logger.error(
                "PostgreSQL retriever creation failed",
                error=str(e),
                exc_info=True,
                component="retriever",
            )
            raise

        # Define signatures for the pipeline
        class CompanyContextGenerator(dspy.Signature):
            """Generate comprehensive context about a company based on its ticker symbol."""

            ticker_symbol = dspy.InputField()
            company_context = dspy.OutputField(
                desc="Detailed context about the company including: 1) Primary business sector and industry, "
                "2) Core business models and revenue streams, 3) Current AI/ML usage and applications, "
                "4) Technology stack and digital infrastructure, 5) Data handling practices, "
                "6) Geographic markets and regulatory jurisdictions"
            )

        class RegulationAnalysis(dspy.Signature):
            """Extract and analyze key regulatory information from documents."""

            question = dspy.InputField(desc="Specific question about AI regulations")
            documents = dspy.InputField(desc="Retrieved regulatory document content")
            key_findings = dspy.OutputField(
                desc="Comprehensive findings including specific regulations, requirements, and standards mentioned"
            )
            scope_of_regulations = dspy.OutputField(
                desc="Detailed scope: which companies, sectors, activities, and use cases are covered"
            )
            compliance_requirements = dspy.OutputField(
                desc="Specific compliance actions required: documentation, assessments, controls, reporting"
            )

        class CompanyImpactAnalysis(dspy.Signature):
            """Analyze regulatory impact on a specific company with detailed reasoning."""

            ticker_symbol = dspy.InputField()
            company_context = dspy.InputField()
            regulatory_findings = dspy.InputField()
            regulatory_scope = dspy.InputField()
            compliance_requirements = dspy.InputField()
            reasoning = dspy.OutputField(
                desc="Detailed step-by-step analysis: 1) How company's business intersects with regulations, "
                "2) Specific regulatory provisions that apply, 3) Compliance gaps and challenges"
            )
            company_impact = dspy.OutputField(
                desc="Concrete impacts: operational changes needed, costs, timeline, competitive effects"
            )
            risk_assessment = dspy.OutputField(
                desc="Risk rating (1-10) with detailed justification covering likelihood, impact, and mitigation difficulty"
            )

        # Enhanced Two-Stage Analysis Pipeline
        class TwoStageAnalyzer(dspy.Module):
            def __init__(self, retriever):
                super().__init__()
                self.retriever = retriever

                # Use Predict instead of ChainOfThought to avoid structured output issues
                self.company_context_generator = dspy.Predict(CompanyContextGenerator)
                self.regulation_analyzer = dspy.Predict(RegulationAnalysis)
                self.company_impact_analyzer = dspy.Predict(CompanyImpactAnalysis)

            def forward(self, ticker_symbol: str, regulation_question: str):
                # Get logger from outer scope
                logger = self.logger if hasattr(self, "logger") else None

                try:
                    if logger:
                        logger.info(
                            "Starting forward pass analysis",
                            ticker=ticker_symbol,
                            question=regulation_question[:50],
                            component="forward",
                        )

                    # Stage 1: Retrieve and analyze regulatory information
                    if logger:
                        logger.debug(
                            "Stage 1: Retrieving regulatory documents",
                            component="forward",
                        )
                    retrieval_result = self.retriever(regulation_question)
                    documents = "\n\n".join(retrieval_result.passages)
                    if logger:
                        logger.debug(
                            "Document retrieval completed",
                            doc_count=len(retrieval_result.passages),
                            component="forward",
                        )

                    if logger:
                        logger.debug(
                            "Stage 1: Analyzing regulatory information",
                            component="forward",
                        )
                    reg_analysis = self.regulation_analyzer(
                        question=regulation_question, documents=documents
                    )
                    if logger:
                        logger.debug(
                            "Regulatory analysis completed", component="forward"
                        )

                    # Generate company context
                    if logger:
                        logger.debug(
                            "Generating company context",
                            ticker=ticker_symbol,
                            component="forward",
                        )
                    context_result = self.company_context_generator(
                        ticker_symbol=ticker_symbol
                    )
                    if logger:
                        logger.debug(
                            "Company context generation completed", component="forward"
                        )

                    # Stage 2: Apply regulatory findings to the specific company
                    if logger:
                        logger.debug(
                            "Stage 2: Analyzing company impact", component="forward"
                        )
                    impact_analysis = self.company_impact_analyzer(
                        ticker_symbol=ticker_symbol,
                        company_context=context_result.company_context,
                        regulatory_findings=reg_analysis.key_findings,
                        regulatory_scope=reg_analysis.scope_of_regulations,
                        compliance_requirements=reg_analysis.compliance_requirements,
                    )
                    if logger:
                        logger.debug(
                            "Company impact analysis completed", component="forward"
                        )

                    if logger:
                        logger.info(
                            "Forward pass completed successfully",
                            ticker=ticker_symbol,
                            component="forward",
                        )

                    return dspy.Prediction(
                        regulation_question=regulation_question,
                        regulatory_findings=reg_analysis.key_findings,
                        company_context=context_result.company_context,
                        reasoning=impact_analysis.reasoning,
                        company_impact=impact_analysis.company_impact,
                        risk_assessment=impact_analysis.risk_assessment,
                        retrieved_passages=retrieval_result.passages,
                    )
                except Exception as e:
                    if logger:
                        logger.error(
                            "Error in forward pass",
                            error=str(e),
                            ticker=ticker_symbol,
                            exc_info=True,
                            component="forward",
                        )
                    else:
                        print(f"Error in forward pass: {e}")
                    # Return a minimal prediction with error info
                    return dspy.Prediction(
                        regulation_question=regulation_question,
                        regulatory_findings=f"Error occurred: {str(e)}",
                        company_context="",
                        reasoning="",
                        company_impact="",
                        risk_assessment="",
                        retrieved_passages=[],
                    )

        # Initialize the analyzer and pass logger reference
        self.analyzer = TwoStageAnalyzer(self.retriever)
        # Pass logger reference to analyzer
        self.analyzer.logger = self.logger

        # Optional: Add a report writer module for better formatting
        class ReportWriter(dspy.Signature):
            """Write a professional regulatory compliance report section."""

            section_title = dspy.InputField()
            content_points = dspy.InputField()
            formatted_section = dspy.OutputField(
                desc="Well-formatted report section with clear structure and professional language"
            )

        self.report_writer = dspy.Predict(ReportWriter)

    def run(
        self, symbol: str, start_date: str = None, end_date: str = None
    ) -> Dict[str, Any]:
        """
        Run method for compatibility with other analyzers.

        Args:
            symbol (str): The ticker symbol of the company to analyze
            start_date (str, optional): Not used for regulatory analysis
            end_date (str, optional): Not used for regulatory analysis

        Returns:
            dict: Regulatory analysis results
        """
        return self.analyze(symbol)

    def analyze(
        self, ticker_symbol: str, custom_regulation_question: str = None
    ) -> Dict[str, Any]:
        """
        Performs a two-stage analysis: first understanding regulations, then applying to the company.

        Args:
            ticker_symbol (str): The ticker symbol of the company to analyze
            custom_regulation_question (str, optional): A specific regulation question

        Returns:
            dict: Results containing analysis for each regulation question and company impact
        """
        if not ticker_symbol:
            raise ValueError("Ticker symbol cannot be empty")

        results = {}

        if custom_regulation_question:
            result = self.analyzer(
                ticker_symbol=ticker_symbol,
                regulation_question=custom_regulation_question,
            )
            return result

        # Analyze all standard regulation questions
        for question in self.regulation_questions:
            try:
                self.logger.info(f"Analyzing: {question[:50]}...", component="analyze")
                result = self.analyzer(
                    ticker_symbol=ticker_symbol, regulation_question=question
                )
                results[question] = {
                    "regulatory_findings": result.regulatory_findings,
                    "company_context": result.company_context,
                    "reasoning": result.reasoning,
                    "company_impact": result.company_impact,
                    "risk_assessment": result.risk_assessment,
                }
            except Exception as e:
                self.logger.error(
                    f"Error analyzing {question}: {e}", component="analyze"
                )
                results[question] = {"error": str(e)}

        return results

    def generate_report(self, ticker_symbol: str) -> str:
        """Generates a comprehensive regulatory impact report for a company."""
        if not ticker_symbol:
            raise ValueError("Ticker symbol cannot be empty")

        try:
            results = self.analyze(ticker_symbol)
        except Exception as e:
            return f"Error generating report: {str(e)}"

        # Start building the report
        report_sections = []

        # Title section
        report_sections.append(f"# Regulatory Impact Analysis for {ticker_symbol}")

        # Company profile section
        if results:
            first_result = next(iter(results.values()))
            if "company_context" in first_result and first_result["company_context"]:
                try:
                    company_section = self.report_writer(
                        section_title="Company Profile",
                        content_points=first_result["company_context"],
                    )
                    report_sections.append(
                        f"\n## Company Profile\n{company_section.formatted_section}"
                    )
                except Exception as e:
                    report_sections.append(
                        f"\n## Company Profile\n{first_result['company_context']}"
                    )

        # Regulatory impact sections
        report_sections.append("\n## Regulatory Impact Assessment")

        for question, result in results.items():
            if "error" not in result:
                try:
                    impact_section = self.report_writer(
                        section_title=question,
                        content_points=f"Findings: {result['regulatory_findings']}\n\nImpact: {result['company_impact']}\n\nRisk: {result['risk_assessment']}",
                    )
                    report_sections.append(
                        f"\n### {question}\n{impact_section.formatted_section}"
                    )
                except Exception as e:
                    # Fallback to direct content if report writer fails
                    report_sections.append(
                        f"\n### {question}\n"
                        f"**Findings:** {result['regulatory_findings']}\n\n"
                        f"**Impact:** {result['company_impact']}\n\n"
                        f"**Risk:** {result['risk_assessment']}"
                    )
            else:
                report_sections.append(
                    f"\n### {question}\n**Error:** {result['error']}"
                )

        return "\n".join(report_sections)

    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()


# Example Usage
if __name__ == "__main__":
    # Setup logging and config
    logger = LoggerManager(name="regulatory_main", level="DEBUG")
    config = Config()

    logger.info("Starting Regulatory Analyzer with PostgreSQL", component="main")

    # You can test different LLM providers by uncommenting these:
    config.set_openrouter_config(
        "openrouter/deepseek/deepseek-chat-v3-0324:free"
    )  # OpenRouter - Let's try this first
    # config.set_ollama_config("ollama/qwen2.5-coder:3b")  # Default Ollama
    # config.set_nvidia_nim_config("nvidia/usdcode-llama-3.1-70b-instruct")  # Nvidia NIM

    try:
        logger.info("Initializing RegulatoryAnalyzer", component="main")
        analyzer = RegulatoryAnalyzer(config=config, logger=logger)

        # Test with a simple question first
        ticker_symbol = "NVDA"
        test_question = "What are the key requirements of recent AI regulations?"

        logger.info(
            "Starting analysis test",
            ticker=ticker_symbol,
            question=test_question,
            component="main",
        )

        result = analyzer.analyze(ticker_symbol, test_question)

        logger.info("Analysis completed", component="main")

        # Print results
        print("\n" + "=" * 60)
        print("REGULATORY ANALYSIS RESULTS")
        print("=" * 60)

        if hasattr(result, "regulatory_findings"):
            print(f"\nREGULATORY FINDINGS:")
            print(result.regulatory_findings)

        if hasattr(result, "company_context"):
            print(f"\nCOMPANY CONTEXT:")
            print(result.company_context)

        if hasattr(result, "company_impact"):
            print(f"\nCOMPANY IMPACT:")
            print(result.company_impact)

        if hasattr(result, "risk_assessment"):
            print(f"\nRISK ASSESSMENT:")
            print(result.risk_assessment)

    except Exception as e:
        logger.error(
            "Error running analyzer", error=str(e), exc_info=True, component="main"
        )
        print(f"\nError running analyzer: {e}")
        print("\nTroubleshooting:")
        print("1. Check if PostgreSQL is running and ai_law_db exists")
        print("2. Verify database connection parameters in .env file")
        print("3. Check if Ollama is running: ollama serve")
        print("4. Verify models are installed: ollama list")
        print("5. Try different LLM provider by uncommenting config lines above")
        print("6. Check logs directory for detailed error information")
