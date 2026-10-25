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
WORKSPACE_TEXT_INDEX_ENABLED = os.getenv('WORKSPACE_TEXT_INDEX_ENABLED', 'true').lower() == 'true'
# Extracted UTF-8 payload per workspace, not a hard cap on SQLite file size.
WORKSPACE_TEXT_INDEX_MAX_BYTES = int(os.getenv('WORKSPACE_TEXT_INDEX_MAX_BYTES', str(64 * 1024 * 1024)))
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
# Document similarity is a retrieval hint; a conservative threshold limits drift.
SEMANTIC_GRAPH_SIMILARITY = max(0.0, min(1.0, float(os.getenv('SEMANTIC_GRAPH_SIMILARITY', '0.82'))))
SEMANTIC_GRAPH_NEIGHBORS = max(1, min(6, int(os.getenv('SEMANTIC_GRAPH_NEIGHBORS', '3'))))
GRAPH_RETRIEVAL_ENABLED = os.getenv('GRAPH_RETRIEVAL_ENABLED', 'true').lower() == 'true'

# Answer verification is optional; long answers use bounded structured batches.
VERIFIER_ENABLED = os.getenv('VERIFIER_ENABLED', 'true').strip().casefold() in {'1', 'true', 'yes', 'on'}
ANSWER_REVIEW_MODE = os.getenv('ANSWER_REVIEW_MODE', 'contextual').strip().casefold()
if ANSWER_REVIEW_MODE not in {'contextual', 'strict'}:
    raise ValueError('ANSWER_REVIEW_MODE must be contextual or strict')
# In strict mode, only these knowledge routes use automatic claim verification.
# Contextual review applies to ordinary sourced chat; /agent/verify stays strict.
VERIFIER_AUTO_STRATEGIES = frozenset(
    value.strip().upper() for value in os.getenv(
        'VERIFIER_AUTO_STRATEGIES',
        'FACT_LOOKUP,DOCUMENT_COMPARISON,DEEP_ANALYSIS,GRAPH_QUERY',
    ).split(',') if value.strip()
)
VERIFIER_CONFIDENCE_THRESHOLD = float(os.getenv('VERIFIER_CONFIDENCE_THRESHOLD', '0.55'))
MAX_VERIFICATION_RETRIES = int(os.getenv('MAX_VERIFICATION_RETRIES', '1'))
EVIDENCE_AUDIT_BATCH_SIZE = max(1, min(2, int(os.getenv('EVIDENCE_AUDIT_BATCH_SIZE', '2'))))
EVIDENCE_AUDIT_COMPACT = os.getenv('EVIDENCE_AUDIT_COMPACT', 'true').lower() == 'true'
SHORT_TASK_SECONDS = max(5, min(90, float(os.getenv('SHORT_TASK_SECONDS', '45'))))
LONG_TASK_SECONDS = max(120, min(540, float(os.getenv('LONG_TASK_SECONDS', '540'))))
CONVERSATION_COMPACTION_ENABLED = os.getenv('CONVERSATION_COMPACTION_ENABLED', 'true').lower() == 'true'
VERIFIER_RETRIEVAL_WEIGHT = float(os.getenv('VERIFIER_RETRIEVAL_WEIGHT', '0.30'))
VERIFIER_CLAIM_SUPPORT_WEIGHT = float(os.getenv('VERIFIER_CLAIM_SUPPORT_WEIGHT', '0.50'))
VERIFIER_SOURCE_AGREEMENT_WEIGHT = float(os.getenv('VERIFIER_SOURCE_AGREEMENT_WEIGHT', '0.20'))
