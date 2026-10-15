"""Deployment profiles; business logic never receives a model filename."""
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from settings import DATA_DIR, API_DIR


def configured_path(name, default):
    value = Path(os.getenv(name, str(default))).expanduser()
    return value if value.is_absolute() else DATA_DIR / value


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_path: Path
    alias: str
    context_size: int = 4096
    temperature: float = 0.2
    max_tokens: int = 2048
    gpu_layers: str = 'auto'
    threads: int = 6
    batch_size: int = 256
    thinking: bool = False
    request_seconds: float = 180
    generation_seconds: float = 90


BASE_PROFILE = ModelProfile(
    name='BALANCED', model_path=configured_path('LLAMA_MODEL_PATH', DATA_DIR / 'models/chat.gguf'),
    alias=os.getenv('LLAMA_MODEL_ALIAS', 'assistant'),
    context_size=int(os.getenv('LLAMA_CONTEXT_SIZE', '4096')),
    max_tokens=int(os.getenv('LLAMA_MAX_TOKENS', '2048')),
    gpu_layers=os.getenv('LLAMA_GPU_LAYERS', 'auto'),
    threads=int(os.getenv('LLAMA_THREADS', '6')),
    batch_size=int(os.getenv('LLAMA_BATCH_SIZE', '256')),
    request_seconds=float(os.getenv('LLAMA_BALANCED_REQUEST_SECONDS', '180')),
    generation_seconds=float(os.getenv('LLAMA_GENERATION_SECONDS', '90')),
)
PROFILES = {'BALANCED': BASE_PROFILE,
            'FAST': replace(BASE_PROFILE, name='FAST', max_tokens=1024,
                            request_seconds=float(os.getenv('LLAMA_FAST_REQUEST_SECONDS', '120'))),
            'REASONING': replace(BASE_PROFILE, name='REASONING', thinking=True,
                                 request_seconds=float(os.getenv('LLAMA_REASONING_REQUEST_SECONDS', '240')),
                                 generation_seconds=float(os.getenv('LLAMA_REASONING_GENERATION_SECONDS', '120')))}
# Optional profile overrides are configuration, never loaded from model responses.
if os.getenv('LLAMA_PROFILES_FILE'):
    for name, values in json.loads(Path(os.environ['LLAMA_PROFILES_FILE']).read_text(encoding='utf-8')).items():
        if 'model_path' in values:
            path = Path(values['model_path']).expanduser()
            values['model_path'] = path if path.is_absolute() else DATA_DIR / path
        PROFILES[name] = replace(BASE_PROFILE, name=name, **values)


def chat_model_catalog():
    choices = [
        ('qwen4b', 'Qwen 3.5 · 4B', DATA_DIR / 'models/chat.gguf'),
        ('qwen9b', 'Qwen 3.5 · 9B', DATA_DIR / 'models/Qwen3.5-9B-Q4_K_M.gguf'),
        ('qwen9b-q3', 'Qwen 3.5 · 9B Q3', DATA_DIR / 'models/Qwen3.5-9B-Q3_K_M.gguf'),
        ('gemma4e4b', 'Gemma 4 · E4B', DATA_DIR / 'models/gemma-4-E4B_q4_0-it.gguf'),
    ]
    if BASE_PROFILE.model_path.resolve() not in {path.resolve() for _, _, path in choices}:
        choices.insert(0, ('configured', BASE_PROFILE.model_path.stem, BASE_PROFILE.model_path))
    return choices


def select_chat_model_path(path):
    for name, profile in list(PROFILES.items()):
        PROFILES[name] = replace(profile, model_path=Path(path))


# The persisted choice changes chat weights only; embeddings stay independent.
try:
    saved_choice = json.loads((DATA_DIR / 'chat-model-selection.json').read_text(encoding='utf-8'))['model_id']
    saved_path = next(path for model_id, _, path in chat_model_catalog() if model_id == saved_choice)
    if saved_path.is_file():
        select_chat_model_path(saved_path)
except (OSError, ValueError, KeyError, StopIteration, TypeError):
    pass


def get_profile(name=None):
    name = getattr(name, 'value', name) or os.getenv('LLAMA_DEFAULT_PROFILE', 'BALANCED')
    if name not in PROFILES:
        raise ValueError(f'Unknown model profile: {name}')
    profile = PROFILES[name]
    if min(profile.context_size, profile.max_tokens, profile.threads, profile.batch_size) <= 0:
        raise ValueError('Model profile limits must be positive')
    if any(not math.isfinite(value) or value <= 0
           for value in (profile.request_seconds, profile.generation_seconds)):
        raise ValueError('Model time limits must be finite and positive')
    return profile


SERVER_URL = os.getenv('LLAMA_SERVER_URL', 'http://127.0.0.1:8081').rstrip('/')
SERVER_EXECUTABLE = configured_path('LLAMA_SERVER_EXECUTABLE', API_DIR.parent / 'runtime/llama/llama-server.exe')
AUTOSTART = os.getenv('LLAMA_AUTOSTART', 'true').lower() == 'true'
STARTUP_TIMEOUT = float(os.getenv('LLAMA_STARTUP_TIMEOUT', '180'))
REQUEST_TIMEOUT = float(os.getenv('LLAMA_REQUEST_TIMEOUT', '120'))
