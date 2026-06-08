"""Configuration module for the Document Analysis System."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
SAMPLE_DOCS_DIR = PROJECT_ROOT / "sample_documents"

# Azure OpenAI settings
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Document processing settings
MAX_DOCUMENT_SIZE_MB = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "50"))
SUPPORTED_FORMATS = set(
    fmt.strip() for fmt in os.getenv(
        "SUPPORTED_FORMATS",
        ".py,.cobol,.cbl,.cic,.cpy,.mps,.src,.ct1,.jcv,.prv,.docx,.html,.xlsx,.xlsm,.xlsb,.txt"
    ).split(",")
)

# Analysis settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Bulk mode: path(s) to documents directory (comma-separated for multiple)
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "")
DOCUMENTS_PATHS = [
    Path(p.strip()) for p in DOCUMENTS_PATH.split(",") if p.strip()
] if DOCUMENTS_PATH else []

# Model settings
# MODEL_REASONING_EFFORT controls internal chain-of-thought depth for reasoning models
# (o1, o3, o4-mini, etc.).  For non-reasoning models (gpt-4o, gpt-4o-mini) set to "none".
#   none   → 1.0× token overhead  (no reasoning chain)
#   low    → 1.5× token overhead
#   medium → 2.5× token overhead
#   high   → 4.0× token overhead
# The LLM client multiplies every max_tokens value by the overhead factor so that
# callers always express *desired output tokens* and the client handles the budget.
MODEL_REASONING_EFFORT = os.getenv("MODEL_REASONING_EFFORT", "medium")  # none | low | medium | high
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "4000"))

# ---------------------------------------------------------------------------
# RAG retrieval parameters
# Tune these to match your model's context window and desired retrieval depth.
# Larger context windows (e.g. 272 K tokens) can support higher k values.
# ---------------------------------------------------------------------------
RAG_PROCESS_DOC_K      = int(os.getenv("RAG_PROCESS_DOC_K",      "5"))  # process-doc chunks / iter
RAG_WORD_K             = int(os.getenv("RAG_WORD_K",             "6"))  # word/doc chunks / iter
RAG_CODE_K             = int(os.getenv("RAG_CODE_K",             "4"))  # code chunks / iter
RAG_CLUSTER_SUMMARY_K  = int(os.getenv("RAG_CLUSTER_SUMMARY_K",  "3"))  # cluster narrative chunks / iter
RAG_CLUSTER_TECHNICAL_K= int(os.getenv("RAG_CLUSTER_TECHNICAL_K","3"))  # cluster technical-map chunks / iter
RAG_CLUSTER_RAW_K      = int(os.getenv("RAG_CLUSTER_RAW_K",      "8"))  # raw code chunks per expanded cluster
RAG_MAX_ITERATIONS     = int(os.getenv("RAG_MAX_ITERATIONS",      "4"))  # retrieval-filter-check cycles

# ---------------------------------------------------------------------------
# LLM cost tracking
# Set these to the per-million-token prices for your deployed model.
# ---------------------------------------------------------------------------
LLM_INPUT_PPM = float(os.getenv("LLM_INPUT_PPM", "0.15"))   # $ per 1 M input tokens
LLM_OUTPUT_PPM = float(os.getenv("LLM_OUTPUT_PPM", "0.60"))  # $ per 1 M output tokens


def validate_config():
    """Validate that all required configuration is set."""
    if not AZURE_OPENAI_API_KEY:
        raise ValueError("AZURE_OPENAI_API_KEY not set in .env")
    if not AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT not set in .env")
    if not AZURE_OPENAI_DEPLOYMENT_NAME:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME not set in .env")
    return True
