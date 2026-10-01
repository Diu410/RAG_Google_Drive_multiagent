"""Deployment profiles; business logic never receives a model filename."""
import json
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


BASE_PROFILE = ModelProfile(
    name='BALANCED', model_path=configured_path('LLAMA_MODEL_PATH', DATA_DIR / 'models/chat.gguf'),
    alias=os.getenv('LLAMA_MODEL_ALIAS', 'assistant'),
    context_size=int(os.getenv('LLAMA_CONTEXT_SIZE', '4096')),
    max_tokens=int(os.getenv('LLAMA_MAX_TOKENS', '2048')),
    gpu_layers=os.getenv('LLAMA_GPU_LAYERS', 'auto'),
    threads=int(os.getenv('LLAMA_THREADS', '6')),
    batch_size=int(os.getenv('LLAMA_BATCH_SIZE', '256')),
)
PROFILES = {'BALANCED': BASE_PROFILE,
            'FAST': replace(BASE_PROFILE, name='FAST', max_tokens=1024),
            'REASONING': replace(BASE_PROFILE, name='REASONING', thinking=True)}
# Optional profile overrides are configuration, never loaded from model responses.
if os.getenv('LLAMA_PROFILES_FILE'):
    for name, values in json.loads(Path(os.environ['LLAMA_PROFILES_FILE']).read_text(encoding='utf-8')).items():
        if 'model_path' in values:
            path = Path(values['model_path']).expanduser()
            values['model_path'] = path if path.is_absolute() else DATA_DIR / path
        PROFILES[name] = replace(BASE_PROFILE, name=name, **values)


def get_profile(name=None):
    name = getattr(name, 'value', name) or os.getenv('LLAMA_DEFAULT_PROFILE', 'BALANCED')
    if name not in PROFILES:
        raise ValueError(f'Unknown model profile: {name}')
    profile = PROFILES[name]
    if min(profile.context_size, profile.max_tokens, profile.threads, profile.batch_size) <= 0:
        raise ValueError('Model profile limits must be positive')
    return profile


SERVER_URL = os.getenv('LLAMA_SERVER_URL', 'http://127.0.0.1:8081').rstrip('/')
SERVER_EXECUTABLE = configured_path('LLAMA_SERVER_EXECUTABLE', API_DIR.parent / 'runtime/llama/llama-server.exe')
AUTOSTART = os.getenv('LLAMA_AUTOSTART', 'true').lower() == 'true'
STARTUP_TIMEOUT = float(os.getenv('LLAMA_STARTUP_TIMEOUT', '180'))
REQUEST_TIMEOUT = float(os.getenv('LLAMA_REQUEST_TIMEOUT', '120'))
