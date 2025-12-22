import uvicorn
import asyncio
import json
import psutil
import pynvml
import base64
import io
from pathlib import Path
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. НАСТРОЙКИ И ПУТИ
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Параметры памяти для RTX 3050 (4GB VRAM) + R7 6800H (16GB RAM)
GPU_MEMORY_MB = 3800  # Оставляем 200MB запаса
RAM_MEMORY_MB = 12000  # Оставляем 4GB для ОС

# Модели
QWEN_MODEL_PATH = MODELS_DIR / "Qwen3-4B-UD-Q5_K_XL.gguf"
TURBO_MODEL_PATH = MODELS_DIR / "z_image_turbo-Q8_0.gguf"

# ==========================================
# 2. КЛАССЫ КОНФИГУРАЦИИ
# ==========================================
class GenerationParams(BaseModel):
    prompt: str
    model: str = "turbo"  # "turbo" или "qwen"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    temperature: float = 0.7
    seed: Optional[int] = None
    width: int = 512
    height: int = 512
    batch_size: int = 1

class SystemStats(BaseModel):
    gpu_memory_free_mb: int
    gpu_memory_used_mb: int
    gpu_memory_total_mb: int
    ram_free_mb: int
    ram_used_mb: int
    ram_total_mb: int
    gpu_utilization: float

# ==========================================
# 3. ИНИЦИАЛИЗАЦИЯ СИСТЕМ
# ==========================================
print("🚀 Инициализация ImGen сервера...")
print(f"📁 Каталог моделей: {MODELS_DIR}")

# FastAPI приложение
app = FastAPI(title="ImGen - Image Generation Server", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NVIDIA GPU мониторинг
try:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ GPU доступен: {torch.cuda.get_device_name(0)}")
except Exception as e:
    GPU_AVAILABLE = False
    DEVICE = torch.device("cpu")
    print(f"⚠️ GPU недоступен: {e}")

# ==========================================
# 4. ЗАГРУЗКА МОДЕЛЕЙ (С ОПТИМИЗАЦИЕЙ ПАМЯТИ)
# ==========================================

class ModelManager:
    def __init__(self):
        self.qwen_model = None
        self.turbo_model = None
        self.current_model = None
        self.load_status = {}
        
    def load_qwen_model(self):
        """Загружает Qwen3-4B с оптимизацией для GPU"""
        try:
            print("🔄 Загрузка Qwen3-4B модели...")
            self.qwen_model = Llama(
                model_path=str(QWEN_MODEL_PATH),
                n_ctx=2048,
                n_gpu_layers=50,  # Максимум слоев на GPU
                n_threads=8,  # Используем 8 CPU потоков
                verbose=False,
                n_batch=256,  # Размер батча
            )
            self.load_status["qwen"] = "✅ Загружена (Текст)"
            print("✅ Qwen3-4B успешно загружена")
            return True
        except Exception as e:
            self.load_status["qwen"] = f"❌ Ошибка: {str(e)[:50]}"
            print(f"❌ Ошибка загрузки Qwen: {e}")
            return False
    
    def load_turbo_model(self):
        """Загружает Stable Diffusion для генерации изображений - оптимально для RTX 3050"""
        try:
            print("🔄 Загрузка Stable Diffusion v1.5 модели для генерации...")
            
            # Используем легкую Stable Diffusion v1.5 - отлично работает на RTX 3050
            from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
            
            model_id = "runwayml/stable-diffusion-v1-5"
            
            self.turbo_model = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if GPU_AVAILABLE else torch.float32,
                safety_checker=None,  # Отключаем для экономии памяти
                requires_safety_checker=False
            )
            
            # Используем более быстрый scheduler
            self.turbo_model.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.turbo_model.scheduler.config
            )
            
            # Оптимизация для малой памяти
            if GPU_AVAILABLE:
                self.turbo_model.enable_attention_slicing()
                self.turbo_model.enable_vae_slicing()
                # Для дополнительной оптимизации
                try:
                    self.turbo_model.enable_sequential_cpu_offload()
                except:
                    pass
            
            if GPU_AVAILABLE:
                self.turbo_model = self.turbo_model.to("cuda")
            
            self.load_status["turbo"] = "✅ Загружена (Stable Diffusion v1.5)"
            print("✅ Stable Diffusion v1.5 успешно загружена")
            return True
        except Exception as e:
            self.load_status["turbo"] = f"❌ Ошибка: {str(e)[:50]}"
            print(f"❌ Ошибка загрузки SD v1.5: {e}")
            return False
    
    def get_model(self, model_name: str):
        """Возвращает загруженную модель"""
        if model_name == "qwen":
            if not self.qwen_model:
                self.load_qwen_model()
            return self.qwen_model
        elif model_name == "turbo":
            if not self.turbo_model:
                self.load_turbo_model()
            return self.turbo_model
        else:
            raise ValueError(f"Неизвестная модель: {model_name}")
    
    def unload_model(self, model_name: str):
        """Выгружает модель из памяти"""
        if model_name == "qwen" and self.qwen_model:
            del self.qwen_model
            self.qwen_model = None
            self.load_status["qwen"] = "❌ Выгружена"
            torch.cuda.empty_cache()
        elif model_name == "turbo" and self.turbo_model:
            del self.turbo_model
            self.turbo_model = None
            self.load_status["turbo"] = "❌ Выгружена"
            torch.cuda.empty_cache()

# Глобальный менеджер моделей
model_manager = ModelManager()

# Загружаем обе модели при старте
print("\n" + "="*50)
print("ЗАГРУЗКА МОДЕЛЕЙ")
print("="*50)
model_manager.load_qwen_model()
asyncio.sleep(1)  # Небольшая задержка между загрузками
model_manager.load_turbo_model()
print("="*50 + "\n")

# ==========================================
# 5. ФУНКЦИИ МОНИТОРИНГА
# ==========================================

def get_system_stats() -> SystemStats:
    """Получает текущие показатели системы"""
    ram = psutil.virtual_memory()
    
    gpu_free_mb = GPU_MEMORY_MB
    gpu_used_mb = 0
    gpu_util = 0.0
    
    if GPU_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_used_mb = mem_info.used // (1024 * 1024)
            gpu_free_mb = mem_info.free // (1024 * 1024)
            
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
        except:
            pass
    
    return SystemStats(
        gpu_memory_free_mb=gpu_free_mb,
        gpu_memory_used_mb=gpu_used_mb,
        gpu_memory_total_mb=GPU_MEMORY_MB,
        ram_free_mb=int(ram.available // (1024 * 1024)),
        ram_used_mb=int(ram.used // (1024 * 1024)),
        ram_total_mb=int(ram.total // (1024 * 1024)),
        gpu_utilization=gpu_util
    )

def check_memory_available(params: GenerationParams) -> bool:
    """Проверяет достаточность памяти для генерации"""
    stats = get_system_stats()
    
    # Примерные требования памяти (в МБ)
    memory_required = 200  # Базовое потребление
    
    if params.model == "qwen":
        memory_required += 2000  # Qwen требует ~2GB VRAM
    elif params.model == "turbo":
        memory_required += 1500  # Turbo требует ~1.5GB VRAM
    
    memory_required += (params.width * params.height * params.batch_size) // 1024
    
    return stats.gpu_memory_free_mb > memory_required

# ==========================================
# 6. API ENDPOINTS
# ==========================================

@app.get("/health")
async def health_check():
    """Проверка здоровья сервера"""
    return {
        "status": "🟢 OK",
        "timestamp": datetime.now().isoformat(),
        "models": model_manager.load_status,
        "device": str(DEVICE)
    }

@app.get("/status")
async def server_status():
    """Получить подробный статус сервера"""
    stats = get_system_stats()
    return {
        "server": "ImGen Image Generation",
        "models_loaded": {
            "qwen3-4b": model_manager.qwen_model is not None,
            "z_image_turbo": model_manager.turbo_model is not None,
        },
        "system": stats.dict(),
        "device": str(DEVICE),
        "available_models": ["qwen", "turbo"]
    }

@app.post("/generate")
async def generate_image(params: GenerationParams):
    """Генерирует изображение на основе текстового описания"""
    
    # Проверяем память
    if not check_memory_available(params):
        raise HTTPException(
            status_code=507,
            detail={
                "error": "Недостаточно GPU памяти",
                "required_mb": 2000,
                "available_mb": get_system_stats().gpu_memory_free_mb
            }
        )
    
    try:
        print(f"\n🎨 Генерация изображения ({params.model}):")
        print(f"   Промпт: {params.prompt[:50]}...")
        print(f"   Параметры: steps={params.num_inference_steps}, scale={params.guidance_scale}")
        
        if params.model == "turbo":
            # Генерация изображения через LCM пайплайн
            model = model_manager.get_model("turbo")
            
            if model is None:
                raise HTTPException(status_code=503, detail="Модель Stable Diffusion не загружена")
            
            try:
                # Используем Qwen для подготовки текстового описания
                qwen = model_manager.get_model("qwen")
                if qwen:
                    # Улучшаем промпт через Qwen
                    enhanced_prompt = qwen.create_completion(
                        prompt=f"Improve this image description: {params.prompt}",
                        max_tokens=50,
                        temperature=0.3,
                        top_p=0.9,
                    )
                    final_prompt = enhanced_prompt["choices"][0]["text"].strip()
                else:
                    final_prompt = params.prompt
            except:
                final_prompt = params.prompt
            
            print(f"   Улучшенный промпт: {final_prompt[:80]}...")
            
            # Генерируем изображение
            image = model(
                final_prompt,
                num_inference_steps=min(params.num_inference_steps, 30),
                guidance_scale=params.guidance_scale,
                height=params.height,
                width=params.width,
                generator=torch.Generator(device="cuda" if GPU_AVAILABLE else "cpu").manual_seed(
                    params.seed if params.seed else np.random.randint(0, 1000000)
                ) if params.seed else None
            ).images[0]
            
            # Конвертируем в base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            stats = get_system_stats()
            
            return {
                "status": "✅ Успешно",
                "model": "Stable Diffusion v1.5",
                "prompt": params.prompt,
                "enhanced_prompt": final_prompt,
                "image": f"data:image/png;base64,{img_base64}",
                "generation_params": params.dict(),
                "system_stats": stats.dict(),
                "timestamp": datetime.now().isoformat()
            }
            
        elif params.model == "qwen":
            # Генерация текста через Qwen (текст-энкодер)
            model = model_manager.get_model("qwen")
            
            if model is None:
                raise HTTPException(status_code=503, detail="Модель Qwen не загружена")
            
            # Qwen генерирует детальное описание для изображения
            full_prompt = f"""Создай детальное описание изображения на основе этого запроса:
Запрос: {params.prompt}

Описание должно быть подробным, с деталями о:
- Объектах и предметах
- Стиле и настроении
- Освещении и цветах
- Композиции и перспективе

Описание (на английском):"""
            
            output = model.create_completion(
                prompt=full_prompt,
                max_tokens=150,
                temperature=params.temperature,
                top_p=0.9,
                repeat_penalty=1.1,
            )
            
            enhanced_text = output["choices"][0]["text"].strip()
            
            stats = get_system_stats()
            
            return {
                "status": "✅ Успешно",
                "model": "Qwen3-4B (Text Encoder)",
                "prompt": params.prompt,
                "output": enhanced_text,
                "type": "text_generation",
                "note": "Используйте это описание как промпт для модели LCM",
                "generation_params": params.dict(),
                "system_stats": stats.dict(),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail=f"Неизвестная модель: {params.model}")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/stream")
async def generate_image_stream(params: GenerationParams):
    """WebSocket для потоковой генерации изображения"""
    # Заглушка для расширения в будущем
    raise HTTPException(status_code=501, detail="Streaming генерация в разработке")

@app.post("/models/unload/{model_name}")
async def unload_model(model_name: str):
    """Выгружает модель из памяти для освобождения GPU"""
    if model_name not in ["qwen", "turbo"]:
        raise HTTPException(status_code=400, detail=f"Неизвестная модель: {model_name}")
    
    model_manager.unload_model(model_name)
    torch.cuda.empty_cache()
    
    return {
        "status": "✅ Модель выгружена",
        "model": model_name,
        "freed_memory": True,
        "system_stats": get_system_stats().dict()
    }

@app.post("/models/reload/{model_name}")
async def reload_model(model_name: str):
    """Перезагружает модель"""
    if model_name not in ["qwen", "turbo"]:
        raise HTTPException(status_code=400, detail=f"Неизвестная модель: {model_name}")
    
    try:
        model_manager.unload_model(model_name)
        await asyncio.sleep(0.5)
        
        if model_name == "qwen":
            model_manager.load_qwen_model()
        else:
            model_manager.load_turbo_model()
        
        return {
            "status": "✅ Модель перезагружена",
            "model": model_name,
            "model_loaded": model_manager.get_model(model_name) is not None,
            "system_stats": get_system_stats().dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/memory")
async def get_memory_info():
    """Подробная информация о памяти"""
    stats = get_system_stats()
    return {
        "gpu": {
            "total_mb": stats.gpu_memory_total_mb,
            "used_mb": stats.gpu_memory_used_mb,
            "free_mb": stats.gpu_memory_free_mb,
            "utilization_percent": stats.gpu_utilization,
            "available": GPU_AVAILABLE
        },
        "ram": {
            "total_mb": stats.ram_total_mb,
            "used_mb": stats.ram_used_mb,
            "free_mb": stats.ram_free_mb,
            "utilization_percent": (stats.ram_used_mb / stats.ram_total_mb * 100) if stats.ram_total_mb > 0 else 0
        }
    }

# ==========================================
# 7. ВЕБ-UI (HTML/CSS/JS)
# ==========================================

HTML_UI = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ImGen - Image Generation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            animation: fadeIn 0.5s ease-in;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
        }
        
        .card {
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            animation: slideUp 0.5s ease-out;
        }
        
        .card h2 {
            margin-bottom: 20px;
            color: #667eea;
            font-size: 1.5em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        input[type="text"],
        input[type="number"],
        input[type="range"],
        select,
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus,
        input[type="number"]:focus,
        input[type="range"]:focus,
        select:focus,
        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        textarea {
            resize: vertical;
            min-height: 100px;
            font-family: inherit;
        }
        
        .slider-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        input[type="range"] {
            flex: 1;
        }
        
        .slider-value {
            min-width: 60px;
            text-align: right;
            font-weight: 600;
            color: #667eea;
        }
        
        .button-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 20px;
        }
        
        button {
            padding: 12px 20px;
            border: none;
            border-radius: 6px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: #667eea;
            color: white;
            grid-column: 1 / -1;
        }
        
        .btn-primary:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        
        .btn-danger {
            background: #ff6b6b;
            color: white;
        }
        
        .btn-danger:hover {
            background: #ee5a52;
        }
        
        .status-card {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        
        .stat {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }
        
        .stat-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 1.3em;
            font-weight: 600;
            color: #333;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s;
        }
        
        .message {
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
            animation: slideDown 0.3s ease-out;
        }
        
        .message.success {
            background: #d4edda;
            color: #155724;
            border-left: 4px solid #28a745;
        }
        
        .message.error {
            background: #f8d7da;
            color: #721c24;
            border-left: 4px solid #f5c6cb;
        }
        
        .message.info {
            background: #d1ecf1;
            color: #0c5460;
            border-left: 4px solid #bee5eb;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideDown {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .output {
            margin-top: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 6px;
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
        }
        
        .output-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 6px;
            margin: 10px 0;
        }
        
        .output-title {
            font-weight: 600;
            margin-bottom: 10px;
            color: #667eea;
        }
        
        .output-text {
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 0.9em;
            line-height: 1.6;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .tab {
            padding: 12px 20px;
            background: none;
            border: none;
            cursor: pointer;
            font-weight: 600;
            color: #999;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        
        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 ImGen</h1>
            <p>Сервер генерации изображений с использованием AI моделей</p>
        </div>
        
        <div class="grid">
            <!-- Левая колона: Управление генерацией -->
            <div class="card">
                <h2>⚙️ Параметры генерации</h2>
                
                <div class="form-group">
                    <label for="model">Выбор модели:</label>
                    <select id="model">
                        <option value="turbo">Stable Diffusion v1.5 (1.5GB, генерация фото)</option>
                        <option value="qwen">Qwen3-4B (2GB, текст энкодер)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="prompt">Описание изображения:</label>
                    <textarea id="prompt" placeholder="Введите описание того, что вы хотите сгенерировать..."></textarea>
                </div>
                
                <div class="form-group">
                    <label>Количество шагов: <span class="slider-value" id="stepsValue">20</span></label>
                    <div class="slider-group">
                        <input type="range" id="steps" min="1" max="50" value="20">
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Guidance Scale: <span class="slider-value" id="guidanceValue">7.5</span></label>
                    <div class="slider-group">
                        <input type="range" id="guidance" min="1" max="20" step="0.5" value="7.5">
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Температура: <span class="slider-value" id="tempValue">0.7</span></label>
                    <div class="slider-group">
                        <input type="range" id="temperature" min="0" max="2" step="0.1" value="0.7">
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="width">Ширина:</label>
                    <input type="number" id="width" min="256" max="1024" value="512" step="64">
                </div>
                
                <div class="form-group">
                    <label for="height">Высота:</label>
                    <input type="number" id="height" min="256" max="1024" value="512" step="64">
                </div>
                
                <div class="form-group">
                    <label for="seed">Seed (оставьте пусто для случайного):</label>
                    <input type="number" id="seed" placeholder="Например: 42">
                </div>
                
                <div class="button-group">
                    <button class="btn-primary" onclick="generateImage()">🚀 Сгенерировать</button>
                </div>
                
                <div id="generationMessage"></div>
                <div class="loading" id="generationLoading">
                    <div class="spinner"></div>
                    <p>Генерируется изображение...</p>
                </div>
            </div>
            
            <!-- Правая колона: Статистика и управление -->
            <div>
                <div class="card" style="margin-bottom: 20px;">
                    <h2>📊 Статистика системы</h2>
                    
                    <div class="tabs">
                        <button class="tab active" onclick="switchTab(event, 'gpu')">GPU</button>
                        <button class="tab" onclick="switchTab(event, 'ram')">RAM</button>
                        <button class="tab" onclick="switchTab(event, 'models')">Модели</button>
                    </div>
                    
                    <div id="gpu" class="tab-content active">
                        <div class="status-card">
                            <div class="stat">
                                <div class="stat-label">Использовано</div>
                                <div class="stat-value" id="gpuUsed">-</div>
                                <div class="progress-bar">
                                    <div class="progress-fill" id="gpuProgress"></div>
                                </div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Свободно</div>
                                <div class="stat-value" id="gpuFree">-</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Всего</div>
                                <div class="stat-value" id="gpuTotal">-</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Утилизация</div>
                                <div class="stat-value" id="gpuUtil">-</div>
                            </div>
                        </div>
                    </div>
                    
                    <div id="ram" class="tab-content">
                        <div class="status-card">
                            <div class="stat">
                                <div class="stat-label">Использовано</div>
                                <div class="stat-value" id="ramUsed">-</div>
                                <div class="progress-bar">
                                    <div class="progress-fill" id="ramProgress"></div>
                                </div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Свободно</div>
                                <div class="stat-value" id="ramFree">-</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Всего</div>
                                <div class="stat-value" id="ramTotal">-</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Утилизация</div>
                                <div class="stat-value" id="ramUtil">-</div>
                            </div>
                        </div>
                    </div>
                    
                    <div id="models" class="tab-content">
                        <div style="display: flex; flex-direction: column; gap: 10px;">
                            <div id="qwenStatus" class="stat">
                                <div class="stat-label">Qwen3-4B</div>
                                <div class="stat-value">Проверка...</div>
                            </div>
                            <div id="turboStatus" class="stat">
                                <div class="stat-label">Stable Diffusion v1.5</div>
                                <div class="stat-value">Проверка...</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>🔧 Управление моделями</h2>
                    <button class="btn-secondary" style="width: 100%; margin-bottom: 10px;" onclick="unloadModel('qwen')">Выгрузить Qwen3-4B</button>
                    <button class="btn-secondary" style="width: 100%; margin-bottom: 10px;" onclick="reloadModel('qwen')">Перезагрузить Qwen3-4B</button>
                    <button class="btn-secondary" style="width: 100%; margin-bottom: 10px;" onclick="unloadModel('turbo')">Выгрузить Stable Diffusion</button>
                    <button class="btn-secondary" style="width: 100%;" onclick="reloadModel('turbo')">Перезагрузить Stable Diffusion</button>
                </div>
            </div>
        </div>
        
        <!-- Результаты генерации -->
        <div class="card">
            <h2>📝 Результаты</h2>
            <div id="output"></div>
        </div>
    </div>
    
    <script>
        const API_BASE = "http://localhost:8001";
        
        // Слайдеры
        document.getElementById('steps').addEventListener('input', (e) => {
            document.getElementById('stepsValue').textContent = e.target.value;
        });
        
        document.getElementById('guidance').addEventListener('input', (e) => {
            document.getElementById('guidanceValue').textContent = e.target.value;
        });
        
        document.getElementById('temperature').addEventListener('input', (e) => {
            document.getElementById('tempValue').textContent = parseFloat(e.target.value).toFixed(1);
        });
        
        // Таблетки
        function switchTab(event, tabName) {
            const tabs = document.querySelectorAll('.tab');
            const contents = document.querySelectorAll('.tab-content');
            
            tabs.forEach(tab => tab.classList.remove('active'));
            contents.forEach(content => content.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        
        // Обновление статистики
        async function updateStats() {
            try {
                const response = await fetch(API_BASE + '/system/memory');
                const data = await response.json();
                
                // GPU
                const gpuPercent = (data.gpu.used_mb / data.gpu.total_mb * 100).toFixed(1);
                document.getElementById('gpuUsed').textContent = data.gpu.used_mb + ' MB';
                document.getElementById('gpuFree').textContent = data.gpu.free_mb + ' MB';
                document.getElementById('gpuTotal').textContent = data.gpu.total_mb + ' MB';
                document.getElementById('gpuUtil').textContent = data.gpu.utilization_percent.toFixed(1) + '%';
                document.getElementById('gpuProgress').style.width = gpuPercent + '%';
                
                // RAM
                const ramPercent = data.ram.utilization_percent.toFixed(1);
                document.getElementById('ramUsed').textContent = data.ram.used_mb + ' MB';
                document.getElementById('ramFree').textContent = data.ram.free_mb + ' MB';
                document.getElementById('ramTotal').textContent = data.ram.total_mb + ' MB';
                document.getElementById('ramUtil').textContent = ramPercent + '%';
                document.getElementById('ramProgress').style.width = ramPercent + '%';
            } catch (error) {
                console.error('Ошибка обновления статистики:', error);
            }
        }
        
        // Статус моделей
        async function updateModelStatus() {
            try {
                const response = await fetch(API_BASE + '/health');
                const data = await response.json();
                
                const qwenLoaded = data.models.qwen.includes('✅');
                const turboLoaded = data.models.turbo.includes('✅');
                
                document.getElementById('qwenStatus').innerHTML = `
                    <div class="stat-label">Qwen3-4B</div>
                    <div class="stat-value">${qwenLoaded ? '✅ Загружена' : '❌ Не загружена'}</div>
                `;
                
                document.getElementById('turboStatus').innerHTML = `
                    <div class="stat-label">z_image_turbo</div>
                    <div class="stat-value">${turboLoaded ? '✅ Загружена' : '❌ Не загружена'}</div>
                `;
            } catch (error) {
                console.error('Ошибка проверки моделей:', error);
            }
        }
        
        // Генерация изображения
        async function generateImage() {
            const prompt = document.getElementById('prompt').value;
            if (!prompt.trim()) {
                showMessage('Пожалуйста, введите описание', 'error');
                return;
            }
            
            const params = {
                prompt: prompt,
                model: document.getElementById('model').value,
                num_inference_steps: parseInt(document.getElementById('steps').value),
                guidance_scale: parseFloat(document.getElementById('guidance').value),
                temperature: parseFloat(document.getElementById('temperature').value),
                width: parseInt(document.getElementById('width').value),
                height: parseInt(document.getElementById('height').value),
                seed: document.getElementById('seed').value ? parseInt(document.getElementById('seed').value) : null,
                batch_size: 1
            };
            
            document.getElementById('generationLoading').style.display = 'block';
            document.getElementById('generationMessage').innerHTML = '';
            
            try {
                const response = await fetch(API_BASE + '/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showMessage('✅ Генерация успешна!', 'success');
                    displayOutput(result);
                } else {
                    showMessage(`❌ Ошибка: ${result.detail}`, 'error');
                }
            } catch (error) {
                showMessage(`❌ Ошибка сети: ${error.message}`, 'error');
            } finally {
                document.getElementById('generationLoading').style.display = 'none';
            }
        }
        
        // Выгрузить модель
        async function unloadModel(modelName) {
            try {
                const response = await fetch(API_BASE + `/models/unload/${modelName}`, {
                    method: 'POST'
                });
                const result = await response.json();
                
                if (response.ok) {
                    showMessage(`✅ ${modelName} выгружена`, 'success');
                    updateModelStatus();
                } else {
                    showMessage(`❌ Ошибка: ${result.detail}`, 'error');
                }
            } catch (error) {
                showMessage(`❌ Ошибка сети: ${error.message}`, 'error');
            }
        }
        
        // Перезагрузить модель
        async function reloadModel(modelName) {
            try {
                const response = await fetch(API_BASE + `/models/reload/${modelName}`, {
                    method: 'POST'
                });
                const result = await response.json();
                
                if (response.ok) {
                    showMessage(`✅ ${modelName} перезагружена`, 'success');
                    updateModelStatus();
                } else {
                    showMessage(`❌ Ошибка: ${result.detail}`, 'error');
                }
            } catch (error) {
                showMessage(`❌ Ошибка сети: ${error.message}`, 'error');
            }
        }
        
        // Вспомогательные функции
        function showMessage(text, type) {
            const messageDiv = document.getElementById('generationMessage');
            messageDiv.innerHTML = `<div class="message ${type}">${text}</div>`;
        }
        
        function displayOutput(result) {
            const output = document.getElementById('output');
            let resultHtml = `
                <div class="output">
                    <div class="output-title">📌 Параметры генерации:</div>
                    <div class="output-text">
                        Модель: ${result.model}
                        Промпт: ${result.prompt}
                        Шаги: ${result.generation_params.num_inference_steps}
                        Guidance Scale: ${result.generation_params.guidance_scale}
                        Разрешение: ${result.generation_params.width}x${result.generation_params.height}
                    </div>
            `;
            
            // Если есть изображение, отображаем его
            if (result.image) {
                resultHtml += `
                    <div class="output-title">🖼️ Сгенерированное изображение:</div>
                    <img src="${result.image}" class="output-image" alt="Generated image">
                `;
            }
            
            // Если есть текстовый результат
            if (result.output) {
                resultHtml += `
                    <div class="output-title">📝 Текстовый результат:</div>
                    <div class="output-text">${result.output}</div>
                `;
            }
            
            // Статистика
            resultHtml += `
                    <div class="output-title">💾 Статистика:</div>
                    <div class="output-text">
                        GPU: ${result.system_stats.gpu_memory_used_mb}/${result.system_stats.gpu_memory_total_mb} MB
                        RAM: ${result.system_stats.ram_used_mb}/${result.system_stats.ram_total_mb} MB
                        GPU Util: ${result.system_stats.gpu_utilization.toFixed(1)}%
                    </div>
                </div>
            `;
            
            output.innerHTML = resultHtml;
        }
        
        // Инициализация
        window.addEventListener('load', () => {
            updateStats();
            updateModelStatus();
            setInterval(updateStats, 2000);
            setInterval(updateModelStatus, 5000);
        });
    </script>
</body>
</html>
"""

@app.get("/")
async def root():
    """Главная страница с веб-интерфейсом"""
    return {
        "message": "ImGen Server",
        "info": "Используйте /ui для веб-интерфейса или /docs для API документации"
    }

@app.get("/ui")
async def get_ui():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=HTML_UI)

# ==========================================
# 8. ЗАПУСК СЕРВЕРА
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 ImGen Server запускается...")
    print("="*60)
    print(f"📍 Веб-интерфейс: http://localhost:8001/ui")
    print(f"📚 API документация: http://localhost:8001/docs")
    print(f"🏥 Здоровье сервера: http://localhost:8001/health")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
