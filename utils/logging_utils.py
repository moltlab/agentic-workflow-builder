import logging
import sys
from typing import Dict, Any

class LoggingManager:
    """Manages logging configuration and control"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logging_config = config.get('logging', {})
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging based on the config"""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.logging_config.get('levels', {}).get('root', 'INFO'))
        
        # Clear any existing handlers to prevent duplicates
        root_logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            self.logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ))
        root_logger.addHandler(console_handler)
        
        # Set component-specific log levels and prevent propagation to avoid duplicates
        levels = self.logging_config.get('levels', {})
        for component, level in levels.items():
            if component != 'root':
                component_logger = logging.getLogger(component)
                component_logger.setLevel(level)
                # Prevent propagation to root logger to avoid duplicate logs
                component_logger.propagate = False
                
                # Add handlers directly to component loggers
                if not component_logger.handlers:  # Only add if no handlers exist
                    component_console_handler = logging.StreamHandler(sys.stdout)
                    component_console_handler.setFormatter(logging.Formatter(
                        self.logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    ))
                    component_logger.addHandler(component_console_handler)
    
    def should_show_llm_output(self) -> bool:
        """Check if LLM outputs should be shown"""
        return self.logging_config.get('show_llm_outputs', False)
    
    def get_component_logger(self, component: str) -> logging.Logger:
        """Get a logger for a specific component"""
        logger = logging.getLogger(component)
        
        # If this logger wasn't pre-configured in levels, set it up to prevent propagation
        if component not in self.logging_config.get('levels', {}):
            logger.propagate = False
            
            # Add handlers if none exist
            if not logger.handlers:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(logging.Formatter(
                    self.logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ))
                logger.addHandler(console_handler)
                
                # Set default level
                logger.setLevel(self.logging_config.get('levels', {}).get('root', 'INFO'))
        
        return logger

# Global instance
_logging_manager = None

def init_logging(config: Dict[str, Any]):
    """Initialize logging with the given configuration"""
    global _logging_manager
    _logging_manager = LoggingManager(config)

def should_show_llm_output() -> bool:
    """Check if LLM outputs should be shown"""
    if _logging_manager is None:
        return False
    return _logging_manager.should_show_llm_output()

def get_logger(component: str) -> logging.Logger:
    """Get a logger for a specific component"""
    if _logging_manager is None:
        return logging.getLogger(component)
    return _logging_manager.get_component_logger(component) 