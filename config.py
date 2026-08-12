"""Central config for Aporia. All paths are relative to the project root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
DB_PATH = DATA_DIR / "aporia.db"

# Embedding model (open, small enough to run on CPU quickly).
# Swap for BAAI/bge-large-en-v1.5 later if quality demands it.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models want this prefix on queries (not on passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chunking targets (in words; ~1.3 tokens/word).
CHUNK_MIN_WORDS = 100
CHUNK_MAX_WORDS = 300

# Retrieval
TOP_K = 12

# Stance layer
ANTHROPIC_MODEL = "claude-opus-5"

# Phase 0 corpus: the free-will debate, public-domain (Project Gutenberg).
# stance_hint is metadata only — the LLM judges each passage on its own.
CORPUS = [
    {
        "gutenberg_id": 9662,
        "author": "David Hume",
        "work": "An Enquiry Concerning Human Understanding",
        "note": "Section VIII 'Of Liberty and Necessity' — compatibilism",
    },
    {
        "gutenberg_id": 3800,
        "author": "Baruch Spinoza",
        "work": "Ethics",
        "note": "necessitarian — the free-will feeling is ignorance of causes",
    },
    {
        "gutenberg_id": 26659,
        "author": "William James",
        "work": "The Will to Believe, and Other Essays in Popular Philosophy",
        "note": "'The Dilemma of Determinism' — defends indeterminism",
    },
    {
        "gutenberg_id": 5683,
        "author": "Immanuel Kant",
        "work": "The Critique of Practical Reason",
        "note": "freedom as a postulate of practical reason",
    },
]
