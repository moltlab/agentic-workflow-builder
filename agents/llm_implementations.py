from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Dict
from pydantic import Field, PrivateAttr
import openai



class CustomLLM(BaseChatModel):
    model: str = "default"
    base_url: str = "http://34.125.174.20:40000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.7
    max_tokens: int = 2048
    name: str = "custom"
    extra_body: Optional[Dict] = None

    _client: openai.Client = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = openai.Client(
            base_url=self.base_url,
            api_key=self.api_key
        )

    @property
    def _llm_type(self) -> str:
        return self.name

    @property
    def model_name(self) -> str:
        return self.model

    def _convert_messages(self, messages: List[BaseMessage]):
        return [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in messages
        ]

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        # Prepare the request parameters
        request_params = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Add extra_body only if it's provided
        if self.extra_body:
            request_params["extra_body"] = self.extra_body

        response = self._client.chat.completions.create(**request_params)
        content = response.choices[0].message.content
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )