# base_tool.py
from abc import abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, PrivateAttr
import dspy
from config import Config
from log import LoggerManager


class BaseFinancialTool(BaseModel):
    """Base class for financial analysis tools."""
    
    # Class attributes that define the tool
    name: str = "BaseFinancialTool"
    desc: str = "Base financial analysis tool"
    
    # Private attributes for Pydantic compatibility
    _config: Config = PrivateAttr()
    _logger: LoggerManager = PrivateAttr()

    def __init__(self, config: Optional[Config] = None, logger: Optional[LoggerManager] = None, **kwargs):
        super().__init__(**kwargs)
        self._config = config or Config()
        self._logger = logger or LoggerManager()

    @property
    def logger(self):
        """Convenience property to access the logger directly."""
        return self._logger

    @abstractmethod
    def forward(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process the analysis request.
        
        Args:
            symbol (str): The financial instrument symbol
            start_date (str): Analysis start date
            end_date (str): Analysis end date
            **kwargs: Additional parameters specific to each tool
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        pass

    def validate_inputs(self, symbol: str) -> bool:
        """Validate input parameters."""
        if not symbol or not isinstance(symbol, str):
            self.logger.info(f"Invalid symbol: {symbol}")
            return False
        return True

    def run(self, query: Optional[str] = None, **kwargs) -> str:
        """
        Interface method for DSPy Avatar.
        
        Args:
            query (Optional[str]): Query string in format "symbol=AAPL,start_date=2024-01-01,end_date=2024-12-31"
                               If not provided, parameters must be in kwargs
            **kwargs: Direct parameters (takes precedence over query string)
            
        Returns:
            str: Formatted analysis results
            
        Examples:
            >>> tool.run("symbol=AAPL,start_date=2024-01-01,end_date=2024-12-31")
            >>> tool.run(symbol="AAPL", start_date="2024-01-01", end_date="2024-12-31")
        """
        try:
            # Initialize parameters from kwargs
            params = kwargs.copy()
            
            # Parse query string if provided
            if query:
                for param in query.split(','):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        # Only use query params if not in kwargs
                        if key.strip() not in params:
                            params[key.strip()] = value.strip()
            
            # Extract required parameters
            required_params = ['symbol', 'start_date', 'end_date']
            missing = [p for p in required_params if p not in params]
            if missing:
                return f"Error: Missing required parameters: {', '.join(missing)}"
            
            # Validate symbol
            if not self.validate_inputs(params['symbol']):
                return f"Error: Invalid symbol '{params['symbol']}'"
            
            # Call forward with parameters
            result = self.forward(
                symbol=params.pop('symbol'),
                start_date=params.pop('start_date'),
                end_date=params.pop('end_date'),
                **params  # Pass any remaining parameters
            )
            
            # Format result
            if isinstance(result, dict):
                return "\n".join(
                    f"{k}: {str(v)}" 
                    for k, v in result.items()
                )
            return str(result)
            
        except Exception as e:
            tool_name = getattr(self, 'name', self.__class__.__name__)
            self.logger.error(f"Error in {tool_name} tool: {e}")
            return f"Error in {tool_name}: {str(e)}"





class BaseBlockchainTool(BaseModel):
    """Base class for blockchain analytics tools."""
    name: str = "BaseBlockchainTool"
    desc: str = "Base blockchain analytics tool"

    _config: Config = PrivateAttr()
    _logger: LoggerManager = PrivateAttr()

    def __init__(self, config: Optional[Config] = None, logger: Optional[LoggerManager] = None, **kwargs):
        super().__init__(**kwargs)
        self._config = config or Config()
        self._logger = logger or LoggerManager()

    @property
    def logger(self):
        return self._logger

    @abstractmethod
    def forward(
        self,
        token_id: str,
        chain_name: Optional[str] = None,
        github_repo: Optional[str] = None,
        **kwargs
    ) -> Any:
        pass

    def validate_inputs(self, token_id: str) -> bool:
        if not token_id or not isinstance(token_id, str):
            self.logger.info(f"Invalid token_id: {token_id}")
            return False
        return True

    def run(self, query: Optional[str] = None, **kwargs) -> str:
        try:
            params = kwargs.copy()
            if query:
                for param in query.split(','):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        if key.strip() not in params:
                            params[key.strip()] = value.strip()
            required = ['token_id']
            missing = [p for p in required if p not in params]
            if missing:
                return f"Error: Missing required parameters: {', '.join(missing)}"
            if not self.validate_inputs(params['token_id']):
                return f"Error: Invalid token_id '{params['token_id']}'"
            result = self.forward(
                token_id=params.pop('token_id'),
                chain_name=params.pop('chain_name', None),
                github_repo=params.pop('github_repo', None),
                **params
            )
            return str(result)
        except Exception as e:
            tool_name = getattr(self, 'name', self.__class__.__name__)
            self.logger.error(f"Error in {tool_name} tool: {e}")
            return f"Error in {tool_name}: {str(e)}"
