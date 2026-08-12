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
    # Phase 1 expansion: god, evil, morality, rights, knowledge, justice.
    {
        "gutenberg_id": 4583,
        "author": "David Hume",
        "work": "Dialogues Concerning Natural Religion",
        "note": "the classic debate on God's existence and the problem of evil",
    },
    {
        "gutenberg_id": 4363,
        "author": "Friedrich Nietzsche",
        "work": "Beyond Good and Evil",
        "note": "critique of morality and the good/evil distinction",
    },
    {
        "gutenberg_id": 11224,
        "author": "John Stuart Mill",
        "work": "Utilitarianism",
        "note": "happiness as the foundation of morality",
    },
    {
        "gutenberg_id": 34901,
        "author": "John Stuart Mill",
        "work": "On Liberty",
        "note": "individual liberty and the harm principle",
    },
    {
        "gutenberg_id": 5682,
        "author": "Immanuel Kant",
        "work": "Fundamental Principles of the Metaphysic of Morals",
        "note": "duty, the categorical imperative, objective morality",
    },
    {
        "gutenberg_id": 7370,
        "author": "John Locke",
        "work": "Second Treatise of Government",
        "note": "natural rights, property, consent of the governed",
    },
    {
        "gutenberg_id": 59,
        "author": "René Descartes",
        "work": "Discourse on the Method",
        "note": "the cogito, method of doubt, proof of God",
    },
    {
        "gutenberg_id": 1497,
        "author": "Plato",
        "work": "The Republic",
        "note": "justice, the good, the ideal state",
    },
    {
        "gutenberg_id": 3742,
        "author": "Thomas Paine",
        "work": "The Rights of Man",
        "note": "natural and civil rights of man",
    },
]

# Bare-topic queries are mapped to a canonical contested claim, then debated
# FOR/AGAINST as usual. Unknown topics fall back to the LLM (cached), and
# failing that are treated as claims verbatim.
TOPIC_CLAIMS = {
    "free will": "Humans have free will",
    "god": "God exists",
    "existence of god": "God exists",
    "the existence of god": "God exists",
    "does god exist": "God exists",
    "evil": "The existence of evil is incompatible with a good God",
    "the problem of evil": "The existence of evil is incompatible with a good God",
    "morality": "Morality is objective",
    "ethics": "Morality is objective",
    "good and evil": "Good and evil are objective, not human inventions",
    "human rights": "Humans have natural rights that governments must respect",
    "natural rights": "Humans have natural rights that governments must respect",
    "determinism": "Every event, including human action, is determined by prior causes",
    "personal identity": "The self persists as one thing over time",
    "the self": "The self persists as one thing over time",
    "knowledge": "Certain knowledge is possible",
    "skepticism": "Certain knowledge is possible",
    "justice": "Justice is objectively real, not mere convention",
    "happiness": "Happiness is the highest good",
    "liberty": "Individual liberty may only be limited to prevent harm to others",
    "freedom": "Humans have free will",
}
