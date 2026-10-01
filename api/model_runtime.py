"""Shared provider registry and thin LangChain adapter. No backend-specific API leaks."""
from enum import Enum
from functools import lru_cache
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from llm_contract import LLMProvider, ProviderError
from model_profiles import get_profile, SERVER_URL
from llama_process import LlamaServerProcess
from llama_provider import LlamaCppProvider


class InferenceMode(str, Enum):
    FAST_CHAT = 'FAST_CHAT'
    NORMAL_RAG = 'NORMAL_RAG'
    COMPLEX_REASONING = 'COMPLEX_REASONING'


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    return LlamaCppProvider(manager=LlamaServerProcess(SERVER_URL, get_profile()))


class ProviderChatModel(BaseChatModel):
    provider: Any
    profile: str = 'BALANCED'
    temperature: float = 0.0
    max_tokens: int = 2048
    thinking: bool = False

    @property
    def _llm_type(self):
        return 'assistant-provider'

    def _messages(self, messages):
        roles = {'human': 'user', 'ai': 'assistant', 'system': 'system'}
        return [{'role': roles.get(m.type, m.type), 'content': m.content} for m in messages]

    def _options(self, stop):
        return dict(profile=self.profile, temperature=self.temperature, max_tokens=self.max_tokens,
                    thinking=self.thinking, stop=stop)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = self.provider.generate(self._messages(messages), **self._options(stop))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=result.text))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for delta in self.provider.stream(self._messages(messages), **self._options(stop)):
            if delta.text:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta.text))
                if run_manager:
                    run_manager.on_llm_new_token(delta.text, chunk=chunk)
                yield chunk


def create_chat_model(*, model=None, mode=InferenceMode.NORMAL_RAG, think=None, **kwargs):
    profile = get_profile(model)
    # Context is a server deployment setting; existing chains may specify it,
    # but it must never silently trigger an oversized context per request.
    kwargs.pop('num_ctx', None)
    max_tokens = kwargs.pop('num_predict', profile.max_tokens)
    return ProviderChatModel(provider=get_llm_provider(), profile=profile.name,
        thinking=profile.thinking if think is None else think, max_tokens=max_tokens, **kwargs)
