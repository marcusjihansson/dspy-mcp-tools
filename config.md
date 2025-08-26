# Configuration Guide (config.py + log.py + utility.py)

This document explains how to configure and use the agent system from a new user's perspective. It focuses on two files that ship with this repository:

- `config.py`: Central configuration for models, API settings, caching, and optional integrations
- `log.py`: A small logger helper (`LoggerManager`) you can use across the project

Note about optional modules: Tuning and security integrations are supported by `config.py`, but the concrete implementations are not included in this repository. You can wire your own modules later (details below).

## Table of contents

- Quick start
- What Config does
- Key fields and defaults
- Provider presets
- Logging with log.py (LoggerManager)
- Security (optional, not included here)
- Tuning (optional, not included here)
- Environment variables used by Config
- Troubleshooting
- Utility helpers (utility.py)
- Recommended .gitignore
- Example: Full minimal script

## Quick start

1. Install dependencies

```
pip install dspy-ai python-dotenv structlog
# Optional for Parquet caching of DataFrames used by utility.cache_result
pip install pandas pyarrow
```

2. Create a .env file (do not commit this file)

```
# Model overrides (optional)
MODEL_NAME=
API_BASE=
API_KEY=

# Optional data/market provider keys used by some tools (leave blank if unused)
ALPACA_KEY=
ALPACA_SECRET=
FINLIGHT_API_KEY=
ALPHA_VANTAGE_API_KEY=
```

3. Minimal usage example

```python
from config import Config

# Instantiate
cfg = Config()

# Choose a provider preset (pick ONE):
# cfg.set_ollama_config(model_name="ollama_chat/gemma3:1b")
# cfg.set_lm_studio_config(model_name="lm_studio//your-model")
# cfg.set_openrouter_config(model_name="openrouter/deepseek/deepseek-chat-v3-0324:free")
# cfg.set_nvidia_nim_config(model_name="nvidia/usdcode-llama-3.1-70b-instruct")
# cfg.set_openai_config(model_name="openai/gpt-4o-mini")
# cfg.set_claude_config(model_name="anthropic/claude-3-5-sonnet-20240620")

# Initialize DSPy LM
lm = cfg.setup_dspy_lm()

# Use your DSPy program as usual...
```

If you use a local model (Ollama or LM Studio), structured output is automatically disabled for compatibility. You can also force this behavior by calling `cfg.setup_dspy_lm(disable_structured_output=True)`.

## What Config does

`Config` centralizes:

- LLM and API configuration (model names, base URLs, API keys)
- Logging selection (or you can pass in your own logger)
- Simple cache directory management
- Optional integration hooks for security and tuning

On initialization, `Config` will load environment variables from `.env` using `python-dotenv`.

## Key fields and defaults

- LLM & DSPy
  - `model_name`: default `"ollama_chat/gemma3:1b"`
  - `llm_temperature`: default `0.7`
  - `api_base`: default `"http://localhost:11434"` (Ollama)
  - `api_key`: default empty (local models typically don’t need a key)
  - `setup_dspy_lm(disable_structured_output: bool = False)`: builds and configures `dspy.LM`
  - Local model detection automatically disables structured output for common local setups

- Caching
  - `use_cache`: default `True`
  - `cache_dir`: default `.cache` (created automatically when caching is enabled)

- Logging
  - `log_level`: default `"INFO"`
  - `logger`: optional; if not provided, `Config` creates a basic `logging.Logger`
  - You can pass in `LoggerManager` from `log.py` if you prefer consistent formatting/context

- Feature toggles
  - `tuning_enabled`: default `True` (no-op unless you provide a tuning integration)
  - `evaluation_enabled`: default `True`

- Optional integrations (not included in this repo)
  - `security_config`: dictionary of options for your own security module
  - `security`: instance of your security module (see “Security (optional)” below)
  - `tuning_config`: instance of your tuning configuration (see “Tuning (optional)” below)

- Environment-backed API keys (loaded automatically if present in `.env`)
  - `ALPACA_KEY`, `ALPACA_SECRET`, `FINLIGHT_API_KEY`, `ALPHA_VANTAGE_API_KEY`
  - Model overrides: `MODEL_NAME`, `API_BASE`, `API_KEY`

## Provider presets

Pick one of the following helpers to set common model providers. Each one sets `model_name`, `api_base`, and `api_key` appropriately (when possible):

- `set_ollama_config(model_name: str = "ollama_chat/deepseek-r1:1.5b")`
  - `api_base` → `http://localhost:11434`
  - `api_key` → empty

- `set_lm_studio_config(model_name: str = "lm_studio//your-model-name")`
  - `api_base` → `http://localhost:1234/v1`
  - `api_key` → `lm-studio`

- `set_openrouter_config(model_name: str = "openrouter/deepseek/deepseek-chat-v3-0324:free")`
  - `api_base` → `https://openrouter.ai/api/v1`
  - `api_key` → from `CHATBOT_KEY` env var

- `set_nvidia_nim_config(model_name: str = "nvidia/usdcode-llama-3.1-70b-instruct")`
  - `api_base` → `https://integrate.api.nvidia.com/v1`
  - `api_key` → from `NVIDIA_NIM` env var

- `set_openai_config(model_name: str = "openai/gpt-4o-mini")`
  - `api_base` → `https://api.openai.com/v1`
  - `api_key` → from `OPENAI_API_KEY` env var

- `set_claude_config(model_name: str = "anthropic/claude-3-5-sonnet-20240620")`
  - `api_base` → `https://api.anthropic.com/v1`
  - `api_key` → from `ANTHROPIC_API_KEY` env var

Tip: you can also override `MODEL_NAME`, `API_BASE`, and `API_KEY` in your `.env` without calling these helpers.

## Logging with log.py (LoggerManager)

`log.py` contains `LoggerManager`, a lightweight wrapper around the standard library logger (or structlog). It writes to both the console and `logs/<name>.log`.

Basic usage:

```python
from log import LoggerManager
from config import Config

logger = LoggerManager(name="my_app", level="INFO", user="alice")

cfg = Config(logger=logger._logger)  # pass the underlying stdlib logger to Config
cfg.setup_dspy_lm()

logger.info("Ready to go")
```

LoggerManager options:

- `name`: log channel and file name (default `"financial_agent"`)
- `level`: `"DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL"` (default `"INFO"`)
- `use_structlog`: set to `True` to output via `structlog` (default `False`)
- `capture_warnings`: capture Python warnings into logs (default `True`)
- `user`: label to include in every log line (default your username)

Note: If you don’t need `LoggerManager`, `Config` will create a basic logger on its own when none is provided.

## Security (optional, not included here)

`Config` has optional hooks for a security layer, but the actual implementation is not included in this GitHub repository. You can add your own module (e.g., `security.py`) that provides:

- `get_secure_module(module_class, *args, **kwargs)`
- `sanitize_input(text: str) -> str`
- `validate_output(text: str) -> str`

Then initialize it and attach it to `Config`:

```python
from config import Config
# from security import Security  # your own implementation (not included)

cfg = Config(
    security_config={
        # your custom settings
    }
)

# Example of wiring, if your module exposes a wrapper/manager
# cfg.security = Security(cfg.security_config)

# Usage in code
clean_prompt = cfg.sanitize_input(user_prompt)
safe_output = cfg.validate_output(model_output)
```

If no security module is attached, these methods log a warning and return the original text unchanged.

## Tuning (optional, not included here)

`Config` can hold a `tuning_config` and expose helpers to toggle and query tuning state. The concrete tuner is not shipped with this repo. If you add your own tuner (for example, a `tuning.py` that exports `TuningConfig`/`TuningManager`), you can wire it like this:

```python
from config import Config
# from tuning import TuningConfig, TuningManager  # your own implementation (not included)

# Example (pseudo-code)
# tuning_cfg = TuningConfig(enabled=True, ...)
# cfg = Config(tuning_config=tuning_cfg)
# tuner = TuningManager(tuning_cfg)

# program = tuner.tune_if_needed(program)
# status = cfg.get_tuning_status()
```

If you don’t provide a tuning module, these fields are harmless no-ops.

## Environment variables used by Config

You can set these in `.env` (recommended) or in your OS environment:

- Model & API
  - `MODEL_NAME`: force a specific model name
  - `API_BASE`: override the base URL for your provider
  - `API_KEY`: override the API key for your provider

- Optional provider keys (used by some tools)
  - `ALPACA_KEY`
  - `ALPACA_SECRET`
  - `FINLIGHT_API_KEY`
  - `ALPHA_VANTAGE_API_KEY`

- Provider-specific keys (only for the corresponding helper)
  - `CHATBOT_KEY` (OpenRouter)
  - `NVIDIA_NIM` (NVIDIA NIM)
  - `OPENAI_API_KEY` (OpenAI)
  - `ANTHROPIC_API_KEY` (Anthropic Claude)

Tip: Never commit your `.env` or real API keys to a public repo.

## Troubleshooting

- "Failed to setup DSPy LM":
  - Check that `model_name`, `api_base`, and credentials match your provider
  - For local models, ensure the server is running (Ollama: 11434, LM Studio: 1234)

- Responses look odd or decoding fails with a local model:
  - Try `cfg.setup_dspy_lm(disable_structured_output=True)`

- Cache directory errors:
  - Ensure `.cache` is writable or set `use_cache=False` when constructing `Config`

- Security/tuning references are present but you don’t use them:
  - You can ignore those fields; they’re only active if you attach your own modules

## Utility helpers (utility.py)

This repo includes two general-purpose decorators you can use throughout your codebase. They integrate nicely with Config’s logger and caching options when used on instance methods that have `self.config` or `self.logger`.

1. retry_with_backoff

- What it does: Retries a function on failure with exponential backoff.
- Signature: `@retry_with_backoff(max_retries: int = 3, retry_delay: float = 1.0, logger: Optional[logging.Logger] = None)`
- How logging works:
  - If you pass `logger=...`, it uses that.
  - If used on instance methods, it will try `self.logger` or `self.config.logger`.
  - Otherwise, it uses a module-level logger.
- Usage examples:

```python
from utility import retry_with_backoff
from log import LoggerManager

logger = LoggerManager(name="jobs")._logger

@retry_with_backoff(max_retries=5, retry_delay=2.0, logger=logger)
def flaky_job(x):
    # your code that might raise
    return do_work(x)

class Client:
    def __init__(self, config):
        self.config = config  # contains .logger

    @retry_with_backoff(max_retries=4)
    def call_api(self, payload):
        return self._do_call(payload)
```

2. cache_result

- What it does: Caches function results on disk. Supports JSON, Parquet (for pandas DataFrames), and Pickle as a fallback.
- Signature: `@cache_result(cache_dir: str = ".cache", use_cache: bool = True, logger: Optional[logging.Logger] = None)`
- Cache key: Derived from function name and a hash of args/kwargs; stored under `cache_dir` with extension based on type.
- How it picks format:
  - `pandas.DataFrame` → Parquet (`.parquet`)
  - JSON-serializable objects → `.json`
  - Fallback for complex objects → Pickle (`.pkl`)
- Integration with Config:
  - If used on instance methods with `self.config`, it reads `use_cache` and `cache_dir` from the `Config` instance, overriding decorator parameters.
  - It also picks up `self.config.logger` if you didn’t pass a logger.
- Usage examples:

```python
from utility import cache_result
from config import Config

cfg = Config()  # has use_cache=True and cache_dir=".cache" by default

@cache_result  # no params → uses defaults, can read from self.config if available
def fast_or_cached(x):
    # heavy computation returning dict/list/primitive
    return compute(x)

class DataService:
    def __init__(self, config: Config):
        self.config = config

    @cache_result  # picks up config.use_cache, config.cache_dir, and config.logger
    def load_prices(self, ticker: str) -> pd.DataFrame:
        return download_prices(ticker)
```

Best practices:

- Keep args/kwargs to JSON-serializable primitives or strings where possible for stable cache keys.
- Handle schema changes by bumping a version in your function name or adding a dummy kwarg so the cache key changes.
- When debugging, disable caching via `Config(use_cache=False)` or `@cache_result(use_cache=False)`.
- Parquet note: DataFrame caching uses Parquet by default; install `pandas` and `pyarrow`.

## Recommended .gitignore

Add a `.gitignore` that prevents committing secrets, local logs, and caches:

```
# Secrets
.env

# Logs
logs/

# Caches
.cache/
__pycache__/
*.pkl
*.parquet
*.json
```

## Example: Full minimal script

```python
from config import Config
from log import LoggerManager

# Logging
logger = LoggerManager(name="demo", level="INFO", user="you")

# Config
cfg = Config(logger=logger._logger)
cfg.set_openrouter_config(model_name="openrouter/deepseek/deepseek-chat-v3-0324:free")
lm = cfg.setup_dspy_lm()

# Now build and run your DSPy program...
```
