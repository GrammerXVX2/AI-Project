from pathlib import Path

# Base directory is ai_server/
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
CHATS_DIR = DATA_DIR / "chats"
RAG_DB_DIR = DATA_DIR / "rag_db"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"

# RAG source documents directory.
# Defaulting to STATIC_DIR to match current repo layout, but use this name in code
# so you can later point it to a dedicated folder without refactoring.
RAG_SOURCE_DIR = STATIC_DIR

# Ensure directories exist
CHATS_DIR.mkdir(parents=True, exist_ok=True)
RAG_DB_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
RAG_SOURCE_DIR.mkdir(parents=True, exist_ok=True)

# RAG Settings
RAG_THRESHOLD = 1.5
SUMMARY_CHECKPOINT_SIZE = 25
SUMMARY_MAX_WORDS = 30

# Models Configuration
MODELS_CONFIG = {
    "Orchestrator": {
        "path": str(MODELS_DIR / "Llama-3.2-1B-Instruct-Q6_K_L.gguf"),
        "n_ctx": 1024,
        "n_gpu_layers": -1,
        "type": "router",
    },
    "Summarizer": {
        "path": str(MODELS_DIR / "Llama-3.2-1B-Instruct-Q6_K_L.gguf"),
        "n_ctx": 2048,
        "n_gpu_layers": -1,
        "type": "summary",
    },
    "Coder Agent": {
        "path": str(MODELS_DIR / "qwen2.5-coder-1.5b-instruct-q5_k_m.gguf"),
        "n_ctx": 4096,
        "n_gpu_layers": 15,
        "type": "coder",
    },
    "Chatter Agent": {
        "path": str(MODELS_DIR / "Qwen3-4B-UD-Q5_K_XL.gguf"),
        "n_ctx": 16384,
        "n_gpu_layers": 10,
        "type": "chat",
    },
    "Image Agent(NOT WORKING)": {
        # Relative path for stable-diffusion.cpp compatibility
        "path": "models/z_image_turbo-Q6_K.gguf",
        "n_ctx": 0,
        "n_gpu_layers": -1,
        "type": "image",
    },
}
