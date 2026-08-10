from pathlib import Path

# ==============================
# Base Paths
# ==============================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"

VECTOR_DB_DIR = BASE_DIR / "vector_db"

LOG_DIR = BASE_DIR / "logs"

# ==============================
# Chunking
# ==============================

CHUNK_SIZE = 400

CHUNK_OVERLAP = 100

# ==============================
# Retrieval
# ==============================

TOP_K = 5

SIMILARITY_THRESHOLD = 1.55

# ==============================
# Embedding Model
# ==============================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==============================
# Collection Name
# ==============================

COLLECTION_NAME = "research_papers"

# ==============================
# LLM
# ==============================

GEMINI_MODEL = "gemini-3.1-flash-lite"