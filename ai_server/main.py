import uvicorn
import asyncio
import json
import os
import uuid
import threading
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from monitor import SystemMonitor

# ==========================================
# 1. НАСТРОЙКИ ПУТЕЙ И БАЗЫ
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
CHATS_DIR = BASE_DIR / "chats"
CHATS_DIR.mkdir(exist_ok=True) # Создаем папку для истории чатов

# Инициализация RAG (CPU)
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') 
chroma_client = PersistentClient(path=str(BASE_DIR / "rag_db"))
collection = chroma_client.get_or_create_collection("codebase")

RAG_THRESHOLD = 1.5 # Увеличили порог, чтобы находить больше совпадений (было 1.2)

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ИСТОРИЯ)
# ==========================================

def get_session_file(session_id: str):
    return CHATS_DIR / f"{session_id}.json"

def load_history(session_id: str) -> List[dict]:
    """Загружает историю сообщений из JSON файла"""
    file_path = get_session_file(session_id)
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("messages", [])
        except Exception:
            return []
    return []

def save_history(session_id: str, messages: List[dict]):
    """Сохраняет историю обратно в JSON"""
    file_path = get_session_file(session_id)
    data = {
        "session_id": session_id,
        "last_updated": datetime.now().isoformat(),
        "messages": messages
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def search_rag_smart(query, n_results=1):
    query_emb = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_emb, n_results=n_results, include=["documents", "distances"]
    )
    if not results['documents'][0]: return None
    
    distance = results['distances'][0][0]
    document = results['documents'][0][0]
    print(f"[RAG DEBUG] Dist: {distance}")
    
    if distance > RAG_THRESHOLD: return None
    return document

# ==========================================
# 3. ЗАГРУЗКА МОДЕЛЕЙ
# ==========================================
import gc

print("Загрузка нейросетей... Ждите...")

monitor = SystemMonitor()

MODELS_CONFIG = {
    "orchestrator": {
        "path": str(BASE_DIR / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        "n_ctx": 1024,
        "n_gpu_layers": -1,
        "type": "router"
    },
    "coder_agent": {
        "path": str(BASE_DIR / "models/qwen2.5-coder-1.5b-instruct-q5_k_m.gguf"),
        "n_ctx": 4096,
        "n_gpu_layers": 15,
        "type": "coder"
    },
    "chat_agent": {
        "path": str(BASE_DIR / "models/Qwen3-4B-UD-Q5_K_XL.gguf"),
        "n_ctx": 2048,
        "n_gpu_layers": 10,
        "type": "chat"
    }
}

loaded_models = {}

def load_model_instance(name: str, gpu_layers_override: Optional[int] = None):
    if name in loaded_models:
        return loaded_models[name]
    
    if name not in MODELS_CONFIG:
        raise ValueError(f"Unknown model: {name}")
        
    config = MODELS_CONFIG[name]
    
    # Determine layers
    layers = config["n_gpu_layers"]
    if gpu_layers_override is not None:
        layers = gpu_layers_override
        
    print(f"[SYSTEM] Loading {name} with {layers} GPU layers...")
    try:
        model = Llama(
            model_path=config["path"],
            n_ctx=config["n_ctx"],
            n_gpu_layers=layers,
            verbose=False
        )
        loaded_models[name] = model
        monitor.set_model_status(name, "idle")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load {name}: {e}")
        return None

def unload_model_instance(name: str):
    if name in loaded_models:
        print(f"[SYSTEM] Unloading {name}...")
        del loaded_models[name]
        gc.collect()
        monitor.set_model_status(name, "unloaded")

# Initialize Monitor with all models (unloaded initially except orchestrator)
for name, config in MODELS_CONFIG.items():
    monitor.register_model(name, None, config)
    monitor.set_model_status(name, "unloaded")

# Load Orchestrator immediately
load_model_instance("orchestrator")

# ==========================================
# 3.5. МОНИТОРИНГ СИСТЕМЫ
# ==========================================
# Логика перенесена в monitor.py

# ==========================================
# 4. API СЕРВЕР
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Модель запроса от вашего UI ---
class UserRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None # Если нет ID, создадим новый
    use_rag: bool = True
    history_limit: int = 5 # Сколько последних сообщений помнить (Context Size)
    max_tokens: int = 512 # Максимальное количество токенов для генерации
    temperature: float = 0.7 # Креативность (0.0 - строго, 1.0 - бред)
    repeat_penalty: float = 1.1 # Штраф за повторения (1.0 - нет штрафа)
    system_prompt: Optional[str] = None # Пользовательский системный промпт

@app.post("/api/models/{name}/load")
async def api_load_model(name: str, mode: str = "gpu"):
    if name not in MODELS_CONFIG:
        raise HTTPException(404, "Model not found")
    
    layers = -1 # Default GPU
    if mode == "cpu":
        layers = 0
    elif mode == "hybrid":
        layers = 14 # Approx 50% for 1.5B/7B models
        
    await asyncio.to_thread(load_model_instance, name, layers)
    return {"status": "loaded", "name": name, "mode": mode}

@app.post("/api/models/{name}/unload")
async def api_unload_model(name: str):
    if name == "orchestrator":
        raise HTTPException(400, "Cannot unload orchestrator")
        
    await asyncio.to_thread(unload_model_instance, name)
    return {"status": "unloaded", "name": name}

@app.post("/api/chat")
async def process_chat(request: UserRequest):
    user_query = request.prompt
    
    # 1. Работа с сессией
    current_session_id = request.session_id or str(uuid.uuid4())
    full_history = load_history(current_session_id)
    
    # Берем только последние N сообщений для контекста (срез)
    context_window = full_history[-request.history_limit:] if full_history else []

    # 2. Оркестрация (классифицируем только текущий запрос, историю слать не надо)
    monitor.set_model_active("orchestrator")
    orc_model = loaded_models.get("orchestrator")
    if not orc_model:
         return {"response": "System Error: Orchestrator not loaded.", "category": "ERROR", "rag_used": False, "session_id": current_session_id}

    orc_prompt = f"<|im_start|>user\nClassify strictly as 'CODING' or 'CHAT'. Query: {user_query}<|im_end|>\n<|im_start|>assistant\n"
    orc_res = await asyncio.to_thread(orc_model.create_completion, orc_prompt, max_tokens=6)
    raw_decision = orc_res['choices'][0]['text'].strip().upper()
    
    # Улучшенная логика определения категории
    decision = "CHAT"
    if any(x in raw_decision for x in ["CODING", "CODE", "PROGRAM", "COURING"]):
        decision = "CODING"
        
    print(f"[ORCHESTRATOR] Raw: {raw_decision} -> {decision} | Session: {current_session_id}")

    rag_context = None

    # 3. Подготовка сообщений для агента
    messages_payload = []
    
    # Добавляем историю в начало payload
    for msg in context_window:
        messages_payload.append({"role": msg["role"], "content": msg["content"]})

    # --- RAG ПОИСК ---
    if request.use_rag:
        rag_context = await asyncio.to_thread(search_rag_smart, user_query)

    # Выбор агента и формирование системного промпта
    if request.system_prompt and request.system_prompt.strip():
        system_msg = request.system_prompt
        target_agent = "coder_agent" if (decision == "CODING" or rag_context) else "chatter_agent"
    else:
        target_agent = "chatter_agent"
        system_msg = "You are a helpful assistant."

        if decision == "CODING" or rag_context:
            target_agent = "coder_agent"
            system_msg = "You are an expert coding assistant and knowledge base expert."

    # Логика фоллбека
    agent_name = target_agent
    if agent_name not in loaded_models:
        alternatives = [k for k in loaded_models.keys() if k != "orchestrator"]
        if alternatives:
            agent_name = alternatives[0]
            print(f"[LOGIC] '{target_agent}' missing. Switching to '{agent_name}'.")
        else:
             return {
                "response": f"⚠️ No active models found. Please load 'coder_agent' or 'chatter_agent'.",
                "category": decision,
                "rag_used": bool(rag_context),
                "session_id": current_session_id
            }
            
    print(f"[LOGIC] Using {agent_name} for {decision} task")
    agent = loaded_models[agent_name]

    # Формируем сообщение с контекстом
    if rag_context:
        current_content = f"Context from knowledge base:\n{rag_context}\n\nUser Query:\n{user_query}"
    else:
        current_content = user_query
        
    final_messages = [{"role": "system", "content": system_msg}] + messages_payload + [{"role": "user", "content": current_content}]

    # Генератор для streaming
    async def generate_stream():
        monitor.set_model_active(agent_name)
        full_response = ""
        
        # Отправляем метаданные сначала
        yield f"data: {json.dumps({'type': 'metadata', 'category': decision, 'rag_used': bool(rag_context), 'model': agent_name, 'session_id': current_session_id})}\n\n"
        
        # Очередь для передачи токенов из треда генерации в async генератор
        token_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def generation_worker():
            try:
                # Блокирующий вызов генерации (запускается в отдельном треде)
                stream = agent.create_chat_completion(
                    messages=final_messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    repeat_penalty=request.repeat_penalty,
                    stream=True
                )
                
                for chunk in stream:
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            # Безопасно кладем в очередь async loop'а
                            asyncio.run_coroutine_threadsafe(token_queue.put(delta['content']), loop)
            except Exception as e:
                print(f"Generation Error: {e}")
            finally:
                # Сигнал завершения
                asyncio.run_coroutine_threadsafe(token_queue.put(None), loop)

        # Запускаем генерацию в отдельном потоке, чтобы не блокировать Event Loop (и WebSocket)
        threading.Thread(target=generation_worker, daemon=True).start()

        try:
            while True:
                # Ждем токен из очереди
                token = await token_queue.get()
                
                if token is None: # Конец генерации
                    break
                
                full_response += token
                yield f"data: {json.dumps({'type': 'content', 'content': token})}\n\n"
            
            # Отправляем сигнал завершения клиенту
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        finally:
            monitor.set_all_idle()
            
            # Сохраняем в историю
            full_history.append({"role": "user", "content": user_query, "timestamp": str(datetime.now())})
            full_history.append({"role": "assistant", "content": full_response, "timestamp": str(datetime.now())})
            await asyncio.to_thread(save_history, current_session_id, full_history)
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")
    #     "session_id": current_session_id, # Возвращаем ID, чтобы UI запомнил его
    #     "model": agent_name
    # }

@app.get("/api/sessions")
async def get_sessions():
    """Возвращает список всех сохраненных сессий (чатов)"""
    sessions = []
    if CHATS_DIR.exists():
        # Сортируем по времени изменения (новые сверху)
        files = sorted(CHATS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    # Берем первое сообщение пользователя как название, или ID
                    title = data.get("messages", [{}])[0].get("content", "New Chat")[:30]
                    sessions.append({
                        "id": data.get("session_id", f.stem),
                        "title": title,
                        "date": data.get("last_updated", "")
                    })
            except:
                continue
    return sessions

@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """Возвращает историю конкретного чата"""
    history = await asyncio.to_thread(load_history, session_id)
    return history

@app.delete("/api/history/{session_id}")
async def delete_history(session_id: str):
    """Удаляет историю конкретного чата"""
    file_path = get_session_file(session_id)
    if file_path.exists():
        try:
            await asyncio.to_thread(os.remove, file_path)
            return {"status": "deleted", "session_id": session_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Запускаем сбор статистики в треде, чтобы не блокировать event loop
            data = await asyncio.to_thread(monitor.get_stats)
            await websocket.send_json(data)
            await asyncio.sleep(1) # Обновление раз в секунду
    except Exception as e:
        print(f"WebSocket disconnected: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)