import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import dspy
from dotenv import load_dotenv


@dataclass
class Config:
    """General configuration for the agent system, extensible to any domain."""

    # LLM & DSPy-related config
    model_name: str = "ollama_chat/gemma3:1b"  # Default to Ollama
    llm_temperature: float = 0.7
    use_cache: bool = True
    cache_dir: str = ".cache"

    # Logging
    log_level: str = "INFO"
    logger: Optional[logging.Logger] = None

    # API keys
    alpaca_key: Optional[str] = None
    alpaca_secret: Optional[str] = None
    finlight_key: Optional[str] = None
    alpha_vantage_key: Optional[str] = None

    # Feature toggles
    tuning_enabled: bool = True
    evaluation_enabled: bool = True

    # Security
    security_config: Dict[str, Any] = field(default_factory=dict)
    security: Optional[Any] = None

    # Domain-specific tuning config (optional and injected externally)
    tuning_config: Optional[Any] = None

    # DSPy-specific configuration
    api_base: str = "http://localhost:11434"  # Default to Ollama
    api_key: str = ""  # Default empty for local models

    def __post_init__(self):
        load_dotenv()

        # Load secrets from env
        self.alpaca_key = os.getenv("ALPACA_KEY", "")
        self.alpaca_secret = os.getenv("ALPACA_SECRET", "")
        self.finlight_key = os.getenv("FINLIGHT_API_KEY", "")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")

        # Override defaults with environment variables if they exist
        self.model_name = os.getenv("MODEL_NAME", self.model_name)
        self.api_base = os.getenv("API_BASE", self.api_base)
        self.api_key = os.getenv("API_KEY", self.api_key)

        # Cache dir
        if self.use_cache:
            Path(self.cache_dir).mkdir(exist_ok=True)

        # Logger - create a simple logger if none provided
        if not self.logger:
            self.logger = self._create_logger()

        # Security (optional imports with error handling)
        if self.security_config:
            try:
                # This would import your security module if it exists
                # from security import SecurityConfig
                # self.security = SecurityConfig(self.security_config)
                self.logger.info(
                    "Security config provided but SecurityConfig not imported"
                )
            except ImportError:
                self.logger.warning(
                    "SecurityConfig not available, security features disabled"
                )

        # Tuning config initialization
        if self.tuning_config and hasattr(self.tuning_config, "initialize"):
            self.tuning_config.initialize(self)

    def _create_logger(self) -> logging.Logger:
        """Create a basic logger if LoggerManager is not available."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, self.log_level.upper()))
        return logger

    def setup_dspy_lm(self, disable_structured_output: bool = False) -> dspy.LM:
        """Setup DSPy language model with current configuration."""
        try:
            # Configure LM parameters based on model type
            lm_params = {
                "model": self.model_name,
                "api_base": self.api_base,
                "api_key": self.api_key,
                "temperature": self.llm_temperature
            }
            
            # Disable structured output for local models that don't support it
            if disable_structured_output or self._is_local_model():
                lm_params["disable_structured_output"] = True
                self.logger.info(f"Disabling structured output for model: {self.model_name}")
            
            lm = dspy.LM(**lm_params)
            dspy.configure(lm=lm)
            self.logger.info(f"DSPy configured with model: {self.model_name}")
            return lm
        except Exception as e:
            self.logger.error(f"Failed to setup DSPy LM: {e}")
            raise
    
    def _is_local_model(self) -> bool:
        """Check if the model is a local model that might not support structured output."""
        local_indicators = [
            "lm_studio",
            "ollama",
            "localhost",
            "127.0.0.1",
            ":1234",  # LM Studio default port
            ":11434"  # Ollama default port
        ]
        
        model_str = f"{self.model_name} {self.api_base}".lower()
        return any(indicator in model_str for indicator in local_indicators)

    def set_ollama_config(self, model_name: str = "ollama_chat/deepseek-r1:1.5b"):
        """Configure for Ollama usage."""
        self.model_name = model_name
        self.api_base = "http://localhost:11434"
        self.api_key = ""
        self.logger.info(f"Configured for Ollama: {model_name}")

    def set_lm_studio_config(self, model_name: str = "lm_studio//your-model-name"):
        """Configure for LM Studio usage."""
        self.model_name = model_name
        self.api_base = "http://localhost:1234/v1"  # LM Studio default
        self.api_key = "lm-studio"  # LM Studio default
        self.logger.info(f"Configured for LM Studio: {model_name}")

    def set_openrouter_config(
        self, model_name: str = "openrouter/deepseek/deepseek-chat-v3-0324:free"
    ):
        """Configure for OpenAI usage."""
        self.model_name = model_name
        self.api_base = "https://openrouter.ai/api/v1"
        self.api_key = os.getenv("CHATBOT_KEY", "")
        self.logger.info(f"Configured for Openrouter: {model_name}")

    def set_nvidia_nim_config(
        self, model_name: str = "nvidia/usdcode-llama-3.1-70b-instruct"
    ):
        """Configure for OpenAI usage."""
        self.model_name = model_name
        self.api_base = "https://integrate.api.nvidia.com/v1"
        self.api_key = os.getenv("NVIDIA_NIM", "")
        self.logger.info(f"Configured for Nvidia NIM: {model_name}")

    def set_openai_config(self, model_name: str = "openai/gpt-4o-mini"):
        """Configure for OpenAI usage."""
        self.model_name = model_name
        self.api_base = "https://api.openai.com/v1"
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.logger.info(f"Configured for OpenAI: {model_name}")

    def set_claude_config(self, model_name: str = "anthropic/claude-3-5-sonnet-20240620"):
        """Configure for Anthropic Claude usage."""
        self.model_name = model_name
        self.api_base = "https://api.anthropic.com/v1"
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.logger.info(f"Configured for Anthropic Claude: {model_name}")

    # ---- Security Utility Methods ----
    def get_secure_module(self, module_class, *args, **kwargs):
        if not self.security:
            raise ValueError("Security system not initialized")
        return self.security.get_secure_module(module_class, *args, **kwargs)

    def sanitize_input(self, input_text):
        if not self.security:
            self.logger.warning(
                "Security system not initialized, returning input as-is"
            )
            return input_text
        return self.security.sanitize_input(input_text)

    def validate_output(self, output_text):
        if not self.security:
            self.logger.warning(
                "Security system not initialized, returning output as-is"
            )
            return output_text
        return self.security.validate_output(output_text)

    # ---- Tuning Management ----
    def enable_tuning(self, enabled: bool = True):
        self.tuning_enabled = enabled
        if self.tuning_config:
            self.tuning_config.enable(enabled)

    def get_tuning_status(self) -> Dict[str, Any]:
        if not self.tuning_config:
            return {"enabled": self.tuning_enabled, "status": "No tuning config set"}
        return self.tuning_config.get_status()


if __name__ == "__main__":
    # Simple DSPy signature for testing
    class BasicQA(dspy.Signature):
        """Answer questions clearly and concisely."""

        question = dspy.InputField()
        answer = dspy.OutputField()

    config = Config()

    config_custom = Config.set_nvidia_nim_config(
        config, model_name="nvidia/usdcode-llama-3.1-70b-instruct"
    )

    lm = config.setup_dspy_lm()

    qa = dspy.ChainOfThought(BasicQA)

    response = qa(question="Name me the countries of the Roman Empire?")

    print(f"✅ Test query successful: {response.answer}")
