
## 🏗️ New Project Structure (Refactored)

The backend has been refactored into a modular architecture:

- **`backend/app/`**: Core application code.
  - **`main.py`**: Entry point.
  - **`config.py`**: Central configuration.
  - **`api/`**: API Routers (`chat`, `models`, `system`).
  - **`services/`**: Business logic (`model_manager`, `rag_service`, `history_manager`).
  - **`utils/`**: Utilities (`system_monitor`).
- **`backend/data/`**: Data storage (`chats`, `rag_db`).

### 🚀 Running the New Server

Run the server using the new entry point:

```powershell
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 🛠️ Services

- **Model Manager**: Handles loading/unloading of LLMs (llama-cpp) and Image Models (stable-diffusion-cpp).
- **RAG Service**: Manages ChromaDB and SentenceTransformers.
- **History Manager**: Manages chat sessions in JSON files.
