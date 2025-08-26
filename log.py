import logging
import structlog
import warnings
from pathlib import Path
import sys
from typing import Any, Dict
from datetime import datetime, timezone


class LoggerManager:
    def __init__(
        self, 
        name: str = "financial_agent", 
        level: str = "INFO", 
        use_structlog: bool = False, 
        capture_warnings: bool = True,
        user: str = "marcusjihansson"  # Default to current user
    ):
        self.name = name
        self._level = level  # Store level as private attribute
        self.use_structlog = use_structlog
        self.capture_warnings = capture_warnings
        self.user = user
        self._logger = self._configure_logger()

    def _configure_logger(self):
        if self.use_structlog:
            return self._configure_structlog_logger()
        else:
            return self._configure_standard_logger()

    def _get_context(self, **additional_context) -> Dict[str, Any]:
        """Get standard logging context with timestamp and user."""
        context = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "user": self.user,
            **additional_context
        }
        return context

    def _configure_standard_logger(self):
        """Configure standard Python logger."""
        log_level = getattr(logging, self._level.upper(), logging.INFO)
        logger = logging.getLogger(self.name)
        
        # Prevent duplicate handlers if logger already exists
        if logger.hasHandlers():
            logger.handlers.clear()
            
        logger.setLevel(log_level)

        # Create a custom formatter that includes user
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - User: %(user)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{self.name}.log")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        if self.capture_warnings:
            logging.captureWarnings(True)
            warnings_logger = logging.getLogger("py.warnings")
            warnings_logger.handlers.clear()
            warnings_logger.addHandler(file_handler)
            warnings_logger.addHandler(stream_handler)

        return logger

    def _configure_structlog_logger(self):
        """Configure structlog logger."""
        log_level = getattr(logging, self._level.upper(), logging.INFO)
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=log_level,
        )

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.contextvars.merge_contextvars,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        if self.capture_warnings:
            def warn_with_structlog(message, category, filename, lineno, file=None, line=None):
                structlog.get_logger("py.warnings").warning(
                    f"{category.__name__}: {message} ({filename}:{lineno})"
                )

            warnings.showwarning = warn_with_structlog

        return structlog.get_logger(self.name)

    def info(self, message: str, **kwargs) -> None:
        """
        Log an info message.
        
        Args:
            message: The message to log
            **kwargs: Additional context parameters
        """
        context = self._get_context(**kwargs)
        if self.use_structlog:
            self._logger.info(message, **context)
        else:
            self._logger.info(message, extra=context)

    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """
        Log an error message.
        
        Args:
            message: The error message to log
            exc_info: Whether to include exception info
            **kwargs: Additional context parameters
        """
        context = self._get_context(**kwargs)
        if self.use_structlog:
            self._logger.error(message, exc_info=exc_info, **context)
        else:
            self._logger.error(message, exc_info=exc_info, extra=context)

    def debug(self, message: str, **kwargs) -> None:
        """
        Log a debug message.
        
        Args:
            message: The debug message to log
            **kwargs: Additional context parameters
        """
        context = self._get_context(**kwargs)
        if self.use_structlog:
            self._logger.debug(message, **context)
        else:
            self._logger.debug(message, extra=context)

    def warning(self, message: str, **kwargs) -> None:
        """
        Log a warning message.
        
        Args:
            message: The warning message to log
            **kwargs: Additional context parameters
        """
        context = self._get_context(**kwargs)
        if self.use_structlog:
            self._logger.warning(message, **context)
        else:
            self._logger.warning(message, extra=context)

    def critical(self, message: str, **kwargs) -> None:
        """
        Log a critical message.
        
        Args:
            message: The critical message to log
            **kwargs: Additional context parameters
        """
        context = self._get_context(**kwargs)
        if self.use_structlog:
            self._logger.critical(message, **context)
        else:
            self._logger.critical(message, extra=context)

    @property
    def level(self) -> str:
        """Get current log level."""
        return self._level

    @level.setter
    def level(self, value: str) -> None:
        """Set log level."""
        self._level = value
        level = getattr(logging, value.upper(), logging.INFO)
        if hasattr(self, '_logger'):
            self._logger.setLevel(level)


if __name__ == "__main__":
    # Create logger with default settings
    logger = LoggerManager(name="my_tool", user="marcusjihansson")

    # Basic logging
    logger.info("Starting process")

    # Logging with additional context
    logger.info("Processing trade", symbol="AAPL", price=150.25)

    # Error logging with exception
    try:
        # some code that might fail
        raise ValueError("Invalid data")
    except Exception as e:
        logger.error("Process failed", exc_info=True, error=str(e))

    # Debug logging with context
    logger.debug("Detailed information", data={"key": "value"})
