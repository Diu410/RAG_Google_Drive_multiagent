"""Paths are relative to the API directory, independently of the launch directory."""
import os
from pathlib import Path
from dotenv import load_dotenv

API_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv('KONTUR_DATA_DIR', str(API_DIR))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(DATA_DIR / '.env')
if DATA_DIR != API_DIR:
    load_dotenv(API_DIR / '.env')
DEFAULT_CHAT_MODEL = os.getenv('LLAMA_DEFAULT_PROFILE', 'BALANCED')
DB_PATH = os.getenv('RAG_DB_PATH', str(DATA_DIR / 'rag_app.db'))
CHROMA_PATH = os.getenv('CHROMA_PATH', str(DATA_DIR / 'chroma_db'))
MAX_UPLOAD_BYTES = int(os.getenv('MAX_UPLOAD_BYTES', str(200 * 1024 * 1024)))
COSYVOICE_ROOT = os.getenv('COSYVOICE_ROOT', '')
COSYVOICE_MODEL_DIR = os.getenv('COSYVOICE_MODEL_DIR', '')
COSYVOICE_PROMPT_AUDIO = os.getenv('COSYVOICE_PROMPT_AUDIO', '')
COSYVOICE_PROMPT_TEXT = os.getenv('COSYVOICE_PROMPT_TEXT', '')
COSYVOICE_SPEAKER = os.getenv('COSYVOICE_SPEAKER', '')
COSYVOICE_SPEED = float(os.getenv('COSYVOICE_SPEED', '1.0'))

# Live-chat defaults favour first-response time and predictable GPU memory use.
FAST_LLM_NUM_CTX = int(os.getenv('LLAMA_CONTEXT_SIZE', '4096'))
FAST_CHAT_NUM_PREDICT = int(os.getenv('FAST_CHAT_NUM_PREDICT', '2048'))
RAG_LLM_NUM_PREDICT = int(os.getenv('RAG_LLM_NUM_PREDICT', '2048'))
COMPLEX_LLM_NUM_PREDICT = int(os.getenv('COMPLEX_LLM_NUM_PREDICT', '3072'))
FAST_RETRIEVAL_K = int(os.getenv('FAST_RETRIEVAL_K', '6'))
FAST_CONTEXT_CHUNKS = int(os.getenv('FAST_CONTEXT_CHUNKS', '3'))

# Deterministic rules keep obvious requests fast. Everything else is classified
# semantically so conversational and creative requests are not forced into RAG.
ROUTER_LLM_FALLBACK = os.getenv('ROUTER_LLM_FALLBACK', 'true').strip().casefold() in {'1', 'true', 'yes', 'on'}

# Graph extraction bounds share the configured model context.
GRAPH_BATCH_CHUNKS = max(1, int(os.getenv('GRAPH_BATCH_CHUNKS', '16')))
GRAPH_BATCH_MAX_CHARS = max(1000, int(os.getenv('GRAPH_BATCH_MAX_CHARS', '9000')))
GRAPH_LLM_NUM_CTX = FAST_LLM_NUM_CTX
GRAPH_LLM_NUM_PREDICT = max(256, int(os.getenv('GRAPH_LLM_NUM_PREDICT', '1400')))
MAX_GRAPH_DEPTH = max(1, min(2, int(os.getenv('MAX_GRAPH_DEPTH', '2'))))
MAX_GRAPH_RESULTS = max(1, int(os.getenv('MAX_GRAPH_RESULTS', '20')))

# Answer verification is optional and uses one batched structured-output call.
VERIFIER_ENABLED = os.getenv('VERIFIER_ENABLED', 'true').strip().casefold() in {'1', 'true', 'yes', 'on'}
# Routine document QA already has a source-only generation prompt.  The more
# expensive claim verifier is reserved for routes where exactness or synthesis
# warrants an additional model pass.  Manual /agent/verify remains explicit.
VERIFIER_AUTO_STRATEGIES = frozenset(
    value.strip().upper() for value in os.getenv(
        'VERIFIER_AUTO_STRATEGIES',
        'FACT_LOOKUP,DOCUMENT_COMPARISON,DEEP_ANALYSIS,GRAPH_QUERY',
    ).split(',') if value.strip()
)
VERIFIER_CONFIDENCE_THRESHOLD = float(os.getenv('VERIFIER_CONFIDENCE_THRESHOLD', '0.55'))
MAX_VERIFICATION_RETRIES = int(os.getenv('MAX_VERIFICATION_RETRIES', '1'))
VERIFIER_RETRIEVAL_WEIGHT = float(os.getenv('VERIFIER_RETRIEVAL_WEIGHT', '0.30'))
VERIFIER_CLAIM_SUPPORT_WEIGHT = float(os.getenv('VERIFIER_CLAIM_SUPPORT_WEIGHT', '0.50'))
VERIFIER_SOURCE_AGREEMENT_WEIGHT = float(os.getenv('VERIFIER_SOURCE_AGREEMENT_WEIGHT', '0.20'))
