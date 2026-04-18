import os
import yaml
from typing import Dict, Any, Optional
from utils.logging_utils import get_logger
from dotenv import load_dotenv

logger = get_logger('config_loader')

load_dotenv()


class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv('CONFIG_PATH', 'config/config.yaml')
        self.config: Optional[Dict[str, Any]] = None

    def load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
            self._override_from_env()
            return self.config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _override_from_env(self) -> None:
        """Override config values from environment variables."""
        assert self.config is not None

        # OpenAI
        if os.getenv('OPENAI_API_KEY'):
            self.config.setdefault('llms', {}).setdefault('openai', {})['api_key'] = os.environ['OPENAI_API_KEY']

        # DeepSeek
        if os.getenv('DEEPSEEK_URL'):
            self.config.setdefault('llms', {}).setdefault('deepseek', {})['endpoint'] = os.environ['DEEPSEEK_URL']

        # Qwen
        if os.getenv('QWEN_URL'):
            self.config.setdefault('llms', {}).setdefault('qwen', {})['endpoint'] = os.environ['QWEN_URL']

        # Database
        if os.getenv('DATABASE_URL'):
            self.config.setdefault('database', {})['url'] = os.environ['DATABASE_URL']

        # MCP
        if os.getenv('MCP_SERVER_URL'):
            self.config.setdefault('mcp', {})['server_url'] = os.environ['MCP_SERVER_URL']
        if os.getenv('MCP_ENABLED'):
            self.config.setdefault('mcp', {})['enabled'] = os.environ['MCP_ENABLED'].lower() == 'true'

        # RAG
        if os.getenv('VECTOR_STORE_PATH'):
            self.config.setdefault('rag', {})['vector_store_path'] = os.environ['VECTOR_STORE_PATH']
        if os.getenv('CHUNK_SIZE'):
            self.config.setdefault('rag', {})['chunk_size'] = int(os.environ['CHUNK_SIZE'])
        if os.getenv('CHUNK_OVERLAP'):
            self.config.setdefault('rag', {})['chunk_overlap'] = int(os.environ['CHUNK_OVERLAP'])
        if os.getenv('EMBEDDING_MODEL'):
            self.config.setdefault('rag', {})['embedding_model'] = os.environ['EMBEDDING_MODEL']

    def reload_config(self) -> Dict[str, Any]:
        self.config = None
        return self.load_config()

    def get_openai_config(self) -> Dict[str, Any]:
        if not self.config:
            self.load_config()
        return self.config.get('llms', {}).get('openai', {})

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        if not self.config:
            self.load_config()
        return self.config.get('agents', {}).get(agent_name, {})
