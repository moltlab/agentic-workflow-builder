"""
Generic LLM Factory for initializing different LLM providers.
Can be used across CrewAI, LangGraph, and other agent frameworks.
Supports vision-capable models (OpenAI gpt-4o, gpt-4-turbo, etc.) for multi-modal inputs.
"""

import os
from typing import Dict, Any, Union, List, Set
from langchain_openai import ChatOpenAI
from agents.llm_implementations import CustomLLM
from utils.logging_utils import get_logger

logger = get_logger('llm_factory')

# OpenAI model IDs/prefixes that support image input (vision). Keep in sync with OpenAI docs.
OPENAI_VISION_MODEL_IDS: Set[str] = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-vision-preview",
    "gpt-4-vision-large",
    "gpt-4o-2024",
    "gpt-4o-2025",
}
OPENAI_VISION_MODEL_PREFIXES: List[str] = [
    "gpt-4o-",
    "gpt-4-turbo",
    "gpt-4-vision",
]


def _openai_model_supports_vision(model_id: str) -> bool:
    """Return True if the given OpenAI model ID supports vision (image input)."""
    if not model_id:
        return False
    model_id_lower = model_id.lower().strip()
    if model_id_lower in OPENAI_VISION_MODEL_IDS:
        return True
    return any(model_id_lower.startswith(p) for p in OPENAI_VISION_MODEL_PREFIXES)


class LLMFactory:
    """Factory class for creating LLM instances across different providers."""
    
    @staticmethod
    def create_llm(config: Dict[str, Any]) -> Union[ChatOpenAI, CustomLLM]:
        """
        Initialize the appropriate LLM based on configuration.
        
        :param config: Dictionary containing LLM configuration
        :return: Initialized LLM instance (ChatOpenAI or CustomLLM)
        """
        llms_config = config.get('llms', {})
        logger.info(f"Initializing LLM with config: {llms_config}")
        
        # Check for OpenAI configuration
        if 'openai' in llms_config:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in config or environment variables")

            openai_cfg = llms_config['openai']
            model_name = openai_cfg.get('model', 'gpt-4')
            model_kwargs: Dict[str, Any] = {}
            image_detail = openai_cfg.get('image_detail')
            if image_detail is not None and image_detail in ('low', 'high', 'auto'):
                model_kwargs['image_detail'] = image_detail
                logger.info(f"OpenAI vision image_detail={image_detail} for model={model_name}")

            return ChatOpenAI(
                api_key=api_key,
                model_name=model_name,
                temperature=openai_cfg.get('temperature', 0.7),
                max_tokens=openai_cfg.get('max_tokens', 2000),
                model_kwargs=model_kwargs or {},
            )
        
        # Check for DeepSeek configuration
        elif 'deepseek' in llms_config:
            DEEPSEEK_URL = os.getenv("DEEPSEEK_URL")
            llm = CustomLLM(
                model=f"deepseek/{llms_config['deepseek'].get('model', 'default')}",
                base_url=llms_config['deepseek'].get('endpoint', DEEPSEEK_URL),
                api_key=llms_config['deepseek'].get('api_key', "EMPTY"),
                temperature=llms_config['deepseek'].get('temperature', 0.3),
                max_tokens=llms_config['deepseek'].get('max_tokens', 1024)
            )
            logger.info(f"Created DeepSeek LLM: {llm}")
            return llm
        
        elif 'qwen' in llms_config:
            QWEN_URL = os.getenv("QWEN_URL")
            llm = CustomLLM(
                name="qwen",
                model=f"openai/Qwen/{llms_config['qwen'].get('model', 'Qwen3-32B')}",
                base_url=llms_config['qwen'].get('endpoint', QWEN_URL),
                api_key=llms_config['qwen'].get('api_key', "None"),
                temperature=llms_config['qwen'].get('temperature', 0.3),
                max_tokens=llms_config['qwen'].get('max_tokens', 1024)
            )
            return llm
        
        # Check for Ollama configuration
        elif 'ollama' in llms_config:
            return ChatOpenAI(
                api_key='EMPTY',
                model_name=llms_config['ollama'].get('model', 'llama2'),
                temperature=llms_config['ollama'].get('temperature', 0.7),
                base_url='http://localhost:11434/v1'
            )
        
        # Check for Local configuration
        elif 'local' in llms_config:
            endpoint = llms_config['local'].get('endpoint')
            if not endpoint:
                raise ValueError("Local endpoint is required in config")
            
            return ChatOpenAI(
                api_key='EMPTY',
                model_name=llms_config['local'].get('model', 'local-model'),
                temperature=llms_config['local'].get('temperature', 0.7),
                base_url=endpoint
            )
        
        else:
            raise ValueError("No valid LLM configuration found in config")
    
    @staticmethod
    def get_llm_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract LLM metadata for logging and tracking.
        
        :param config: Dictionary containing LLM configuration
        :return: Dictionary with LLM metadata
        """
        llms_config = config.get('llms', {})
        
        # Determine provider
        provider = next((k for k in ["openai", "deepseek", "qwen", "ollama", "local"] 
                        if k in llms_config), "unknown")
        
        provider_config = llms_config.get(provider, {})
        
        return {
            "model_provider": provider,
            "model_name": provider_config.get('model', 'unknown'),
            "temperature": provider_config.get('temperature', 0.7),
            "max_tokens": provider_config.get('max_tokens', 2000)
        }

    @staticmethod
    def get_vision_capabilities(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return vision capabilities for the configured LLM (for API validation and UI).

        Reads agent config keys:
          - multimodal.enabled, multimodal.supported_types, multimodal.max_attachments
          - llms.openai.image_detail (OpenAI only)

        Returns:
            supports_images: bool
            supports_video: bool (OpenAI: False for now)
            max_images: int (from multimodal.max_attachments or default)
            supported_formats: list of MIME types (from multimodal.supported_types or default)
            model_id: str (for logging)
        """
        multimodal = config.get('multimodal') or {}
        llms_config = config.get('llms', {})
        provider = next(
            (k for k in ["openai", "deepseek", "qwen", "ollama", "local"] if k in llms_config),
            None,
        )
        provider_config = llms_config.get(provider, {}) if provider else {}
        model_id = provider_config.get('model', '')

        supports_images = False
        supports_video = False

        if provider == 'openai':
            supports_images = _openai_model_supports_vision(model_id)
            # OpenAI does not support native video; would need frame extraction (Layer 8)
            supports_video = False
        # Future: elif provider == 'anthropic': ...
        # Future: elif provider == 'google': supports_video = True ...

        default_image_types = [
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
        ]
        default_max_attachments = 5

        return {
            "supports_images": supports_images,
            "supports_video": supports_video,
            "max_images": multimodal.get('max_attachments', default_max_attachments),
            "supported_formats": multimodal.get('supported_types') or default_image_types,
            "model_id": model_id,
            "multimodal_enabled": multimodal.get('enabled', False),
        }
