import asyncio
import base64
import io
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import pynvml
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from llama_cpp import Llama
from PIL import Image
from pydantic import BaseModel
from stable_diffusion_cpp import StableDiffusion
from transformers import pipeline

warnings.filterwarnings("ignore")

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
FLUX_MODEL_PATH = MODELS_DIR / "flux1-schnell-Q4_K_S.gguf"
# Realistic Vision v5.1 (монолит для diffusers)
REALVISION_MODEL_PATH = MODELS_DIR / "Realistic_Vision_V5.1_fp16-no-ema.safetensors"
REALVISION_VAE_PATH = Path(os.getenv("REALVISION_VAE_PATH", MODELS_DIR / "vae-ft-mse-840000-ema-pruned.safetensors"))
# Опциональные компоненты Flux (если скачаны отдельно)
FLUX_CLIP_L_PATH = os.getenv("FLUX_CLIP_L_PATH", "")
FLUX_CLIP_G_PATH = os.getenv("FLUX_CLIP_G_PATH", "")
FLUX_VAE_PATH = os.getenv("FLUX_VAE_PATH", "")
FLUX_T5XXL_PATH = os.getenv("FLUX_T5XXL_PATH", "")


# ==========================================
# 2. КЛАССЫ КОНФИГУРАЦИИ
# ==========================================
class GenerationParams(BaseModel):
    prompt: str
    model: str = "turbo"  # "turbo" | "qwen" | "flux"
    mode: str = "text2img"  # text2img | img2img | inpaint | upscale
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    temperature: float = 0.7
    seed: Optional[int] = None
    width: int = 512
    height: int = 512
    batch_size: int = 1
    negative_prompt: Optional[str] = None
    strength: float = 0.6  # для img2img/inpaint
    upscale_factor: float = 2.0
    init_image: Optional[str] = None  # base64 data URL
    mask_image: Optional[str] = None  # base64 data URL


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
        # Заполняем статусы по умолчанию, чтобы health отдавал ключи даже без автозагрузки
        self.load_status = {
            "qwen": "⏸ Не загружена",
            "turbo": "⏸ Не загружена",
            "flux": "⏸ Заменена на Realistic Vision",
        }

    def load_qwen_model(self):
        """Загружает Qwen3-4B с оптимизацией для GPU"""
        try:
            print("🔄 Загрузка Qwen3-4B модели...")
 
# ==========================================
# 7. ВЕБ-ИНТЕРФЕЙС
# ==========================================

HTML_UI = """<!DOCTYPE html>
<html lang=\"ru\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>ImGen</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #1f1c2c, #928dab);
            color: #333;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px 60px;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            animation: fadeIn 0.5s ease-in;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { font-size: 1.1em; opacity: 0.9; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        .card { background: white; border-radius: 10px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); animation: slideUp 0.5s ease-out; }
        .card h2 { margin-bottom: 20px; color: #667eea; font-size: 1.5em; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
        input[type="text"], input[type="number"], input[type="range"], select, textarea {
            width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 6px; font-size: 1em; transition: border-color 0.3s;
        }
        input[type="text"]:focus, input[type="number"]:focus, input[type="range"]:focus, select:focus, textarea:focus {
            outline: none; border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        textarea { resize: vertical; min-height: 100px; font-family: inherit; }
        .slider-group { display: flex; gap: 10px; align-items: center; }
        input[type="range"] { flex: 1; }
        .slider-value { min-width: 40px; text-align: right; font-weight: 600; }
        .button-group { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 20px; }
        button { padding: 12px 20px; border: none; border-radius: 6px; font-size: 1em; font-weight: 600; cursor: pointer; transition: all 0.3s; }
        .btn-primary { background: #667eea; color: white; grid-column: 1 / -1; }
        .btn-primary:hover { background: #5568d3; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
        .btn-secondary { background: #f0f0f0; color: #333; }
        .btn-secondary:hover { background: #e0e0e0; }
        .status-card { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
        .stat { background: #f5f5f5; padding: 15px; border-radius: 6px; border-left: 4px solid #667eea; }
        .stat-label { font-size: 0.9em; color: #666; margin-bottom: 5px; }
        .stat-value { font-size: 1.3em; font-weight: 600; color: #333; }
        .progress-bar { width: 100%; height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden; margin-top: 10px; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); transition: width 0.3s; }
        .message { padding: 15px; border-radius: 6px; margin-bottom: 15px; animation: slideDown 0.3s ease-out; }
        .message.success { background: #d4edda; color: #155724; border-left: 4px solid #28a745; }
        .message.error { background: #f8d7da; color: #721c24; border-left: 4px solid #f5c6cb; }
        .loading { display: none; text-align: center; padding: 20px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes slideDown { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
        .output { margin-top: 20px; padding: 15px; background: #f9f9f9; border-radius: 6px; max-height: 600px; overflow-y: auto; border: 1px solid #e0e0e0; }
        .output-image { max-width: 100%; max-height: 400px; border-radius: 6px; margin: 10px 0; }
        .output-title { font-weight: 600; margin-bottom: 10px; color: #667eea; }
        .output-text { white-space: pre-wrap; word-break: break-word; font-size: 0.9em; line-height: 1.6; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid #e0e0e0; }
        .tab { padding: 12px 20px; background: none; border: none; cursor: pointer; font-weight: 600; color: #999; border-bottom: 3px solid transparent; transition: all 0.3s; }
        .tab.active { color: #667eea; border-bottom-color: #667eea; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class=\"container\">
        <div class=\"header\">
            <h1>🎨 ImGen</h1>
            <p>Сервер генерации изображений с использованием AI моделей</p>
        </div>
        <div class=\"grid\">
            <div class=\"card\">
                <h2>⚙️ Параметры генерации</h2>
                <div class=\"form-group\">
                    <label for=\"model\">Выбор модели:</label>
                    <select id=\"model\">
                        <option value=\"turbo\">Realistic Vision v5.1 (diffusers)</option>
                        <option value=\"qwen\">Qwen3-4B (text encoder)</option>
                    </select>
                </div>
                <div class=\"form-group\">
                    <label for=\"prompt\">Описание изображения:</label>
                    <textarea id=\"prompt\" placeholder=\"Введите описание...\"></textarea>
                </div>
                <div class=\"form-group\">
                    <label for=\"negprompt\">Negative prompt (опционально):</label>
                    <textarea id=\"negprompt\" placeholder=\"Что исключить...\"></textarea>
                </div>
                <div class=\"form-group\">
                    <label>Количество шагов: <span class=\"slider-value\" id=\"stepsValue\">20</span></label>
                    <div class=\"slider-group\"><input type=\"range\" id=\"steps\" min=\"1\" max=\"50\" value=\"20\"></div>
                </div>
                <div class=\"form-group\">
                    <label>Guidance Scale: <span class=\"slider-value\" id=\"guidanceValue\">7.5</span></label>
                    <div class=\"slider-group\"><input type=\"range\" id=\"guidance\" min=\"1\" max=\"20\" step=\"0.5\" value=\"7.5\"></div>
                </div>
                <div class=\"form-group\">
                    <label>Температура: <span class=\"slider-value\" id=\"tempValue\">0.7</span></label>
                    <div class=\"slider-group\"><input type=\"range\" id=\"temperature\" min=\"0\" max=\"2\" step=\"0.1\" value=\"0.7\"></div>
                </div>
                <div class=\"form-group\">
                    <label for=\"width\">Ширина:</label>
                    <input type=\"number\" id=\"width\" min=\"256\" max=\"1024\" value=\"512\" step=\"64\">
                </div>
                <div class=\"form-group\">
                    <label for=\"height\">Высота:</label>
                    <input type=\"number\" id=\"height\" min=\"256\" max=\"1024\" value=\"512\" step=\"64\">
                </div>
                <div class=\"form-group\">
                    <label for=\"seed\">Seed (оставьте пусто для случайного):</label>
                    <input type=\"number\" id=\"seed\" placeholder=\"Например: 42\">
                </div>
                <div class=\"button-group\">
                    <button class=\"btn-primary\" onclick=\"generateImage()\">🚀 Сгенерировать</button>
                </div>
                <div id=\"generationMessage\"></div>
                <div class=\"loading\" id=\"generationLoading\">
                    <div class=\"spinner\"></div>
                    <p>Генерируется изображение...</p>
                </div>
            </div>
            <div>
                <div class=\"card\" style=\"margin-bottom: 20px;\">
                    <h2>📊 Статистика системы</h2>
                    <div class=\"tabs\">
                        <button class=\"tab active\" onclick=\"switchTab(event, 'gpu')\">GPU</button>
                        <button class=\"tab\" onclick=\"switchTab(event, 'ram')\">RAM</button>
                        <button class=\"tab\" onclick=\"switchTab(event, 'models')\">Модели</button>
                    </div>
                    <div id=\"gpu\" class=\"tab-content active\">
                        <div class=\"status-card\">
                            <div class=\"stat\">
                                <div class=\"stat-label\">Использовано</div>
                                <div class=\"stat-value\" id=\"gpuUsed\">-</div>
                                <div class=\"progress-bar\"><div class=\"progress-fill\" id=\"gpuProgress\"></div></div>
                            </div>
                            <div class=\"stat\"><div class=\"stat-label\">Свободно</div><div class=\"stat-value\" id=\"gpuFree\">-</div></div>
                            <div class=\"stat\"><div class=\"stat-label\">Всего</div><div class=\"stat-value\" id=\"gpuTotal\">-</div></div>
                            <div class=\"stat\"><div class=\"stat-label\">Утилизация</div><div class=\"stat-value\" id=\"gpuUtil\">-</div></div>
                        </div>
                    </div>
                    <div id=\"ram\" class=\"tab-content\">
                        <div class=\"status-card\">
                            <div class=\"stat\"><div class=\"stat-label\">Использовано</div><div class=\"stat-value\" id=\"ramUsed\">-</div><div class=\"progress-bar\"><div class=\"progress-fill\" id=\"ramProgress\"></div></div></div>
                            <div class=\"stat\"><div class=\"stat-label\">Свободно</div><div class=\"stat-value\" id=\"ramFree\">-</div></div>
                            <div class=\"stat\"><div class=\"stat-label\">Всего</div><div class=\"stat-value\" id=\"ramTotal\">-</div></div>
                            <div class=\"stat\"><div class=\"stat-label\">Утилизация</div><div class=\"stat-value\" id=\"ramUtil\">-</div></div>
                        </div>
                    </div>
                    <div id=\"models\" class=\"tab-content\">
                        <div style=\"display: flex; flex-direction: column; gap: 10px;\">
                            <div id=\"qwenStatus\" class=\"stat\"><div class=\"stat-label\">Qwen3-4B</div><div class=\"stat-value\">Проверка...</div></div>
                            <div id=\"turboStatus\" class=\"stat\"><div class=\"stat-label\">Realistic Vision v5.1</div><div class=\"stat-value\">Проверка...</div></div>
                        </div>
                    </div>
                </div>
                <div class=\"card\">
                    <h2>🔧 Управление моделями</h2>
                    <button class=\"btn-secondary\" style=\"width: 100%; margin-bottom: 10px;\" onclick=\"unloadModel('qwen')\">Выгрузить Qwen3-4B</button>
                    <button class=\"btn-secondary\" style=\"width: 100%; margin-bottom: 10px;\" onclick=\"reloadModel('qwen')\">Перезагрузить Qwen3-4B</button>
                    <button class=\"btn-secondary\" style=\"width: 100%; margin-bottom: 10px;\" onclick=\"unloadModel('turbo')\">Выгрузить Realistic Vision</button>
                    <button class=\"btn-secondary\" style=\"width: 100%;\" onclick=\"reloadModel('turbo')\">Перезагрузить Realistic Vision</button>
                </div>
            </div>
        </div>
        <div class=\"card\">
            <h2>📝 Результаты</h2>
            <div id=\"output\"></div>
        </div>
    </div>
    <script>
        const API_BASE = "http://localhost:8001";
        document.getElementById('steps').addEventListener('input', (e) => { document.getElementById('stepsValue').textContent = e.target.value; });
        document.getElementById('guidance').addEventListener('input', (e) => { document.getElementById('guidanceValue').textContent = e.target.value; });
        document.getElementById('temperature').addEventListener('input', (e) => { document.getElementById('tempValue').textContent = parseFloat(e.target.value).toFixed(1); });
        function switchTab(event, tabName) {
            const tabs = document.querySelectorAll('.tab');
            const contents = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.classList.remove('active'));
            contents.forEach(content => content.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        async function updateStats() {
            try {
                const response = await fetch(API_BASE + '/system/memory');
                const data = await response.json();
                const gpuPercent = (data.gpu.used_mb / data.gpu.total_mb * 100).toFixed(1);
                document.getElementById('gpuUsed').textContent = data.gpu.used_mb + ' MB';
                document.getElementById('gpuFree').textContent = data.gpu.free_mb + ' MB';
                document.getElementById('gpuTotal').textContent = data.gpu.total_mb + ' MB';
                document.getElementById('gpuUtil').textContent = data.gpu.utilization_percent.toFixed(1) + '%';
                document.getElementById('gpuProgress').style.width = gpuPercent + '%';
                const ramPercent = data.ram.utilization_percent.toFixed(1);
                document.getElementById('ramUsed').textContent = data.ram.used_mb + ' MB';
                document.getElementById('ramFree').textContent = data.ram.free_mb + ' MB';
                document.getElementById('ramTotal').textContent = data.ram.total_mb + ' MB';
                document.getElementById('ramUtil').textContent = ramPercent + '%';
                document.getElementById('ramProgress').style.width = ramPercent + '%';
            } catch (error) { console.error('Ошибка обновления статистики:', error); }
        }
        async function updateModelStatus() {
            try {
                const response = await fetch(API_BASE + '/health');
                const data = await response.json();
                const qwenLoaded = data.models.qwen.includes('✅');
                const turboLoaded = data.models.turbo.includes('✅');
                document.getElementById('qwenStatus').innerHTML = `<div class="stat-label">Qwen3-4B</div><div class="stat-value">${qwenLoaded ? '✅ Загружена' : '❌ Не загружена'}</div>`;
                document.getElementById('turboStatus').innerHTML = `<div class="stat-label">Realistic Vision v5.1</div><div class="stat-value">${turboLoaded ? '✅ Загружена' : '❌ Не загружена'}</div>`;
            } catch (error) { console.error('Ошибка проверки моделей:', error); }
        }
        async function generateImage() {
            const prompt = document.getElementById('prompt').value;
            if (!prompt.trim()) { showMessage('Пожалуйста, введите описание', 'error'); return; }
            const model = document.getElementById('model').value;
            const negative = document.getElementById('negprompt').value;
            const params = {
                prompt: prompt,
                negative_prompt: negative || null,
                model: model,
                mode: 'text2img',
                num_inference_steps: parseInt(document.getElementById('steps').value),
                guidance_scale: parseFloat(document.getElementById('guidance').value),
                temperature: parseFloat(document.getElementById('temperature').value),
                width: parseInt(document.getElementById('width').value),
                height: parseInt(document.getElementById('height').value),
                seed: document.getElementById('seed').value ? parseInt(document.getElementById('seed').value) : null,
                batch_size: 1,
                strength: null,
                upscale_factor: null,
                init_image: null,
                mask_image: null
            };
            document.getElementById('generationLoading').style.display = 'block';
            document.getElementById('generationMessage').innerHTML = '';
            try {
                const response = await fetch(API_BASE + '/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params) });
                const result = await response.json();
                if (response.ok) { showMessage('✅ Генерация успешна!', 'success'); displayOutput(result); }
                else { showMessage(`❌ Ошибка: ${result.detail}`, 'error'); }
            } catch (error) { showMessage(`❌ Ошибка сети: ${error.message}`, 'error'); }
            finally { document.getElementById('generationLoading').style.display = 'none'; }
        }
        async function unloadModel(modelName) {
            try {
                const response = await fetch(API_BASE + `/models/unload/${modelName}`, { method: 'POST' });
                const result = await response.json();
                if (response.ok) { showMessage(`✅ ${modelName} выгружена`, 'success'); updateModelStatus(); }
                else { showMessage(`❌ Ошибка: ${result.detail}`, 'error'); }
            } catch (error) { showMessage(`❌ Ошибка сети: ${error.message}`, 'error'); }
        }
        async function reloadModel(modelName) {
            try {
                const response = await fetch(API_BASE + `/models/reload/${modelName}`, { method: 'POST' });
                const result = await response.json();
                if (response.ok) { showMessage(`✅ ${modelName} перезагружена`, 'success'); updateModelStatus(); }
                else { showMessage(`❌ Ошибка: ${result.detail}`, 'error'); }
            } catch (error) { showMessage(`❌ Ошибка сети: ${error.message}`, 'error'); }
        }
        function showMessage(text, type) { document.getElementById('generationMessage').innerHTML = `<div class="message ${type}">${text}</div>`; }
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
            if (result.image) { resultHtml += `<div class="output-title">🖼️ Сгенерированное изображение:</div><img src="${result.image}" class="output-image" alt="Generated image">`; }
            if (result.output) { resultHtml += `<div class="output-title">📝 Текстовый результат:</div><div class="output-text">${result.output}</div>`; }
            resultHtml += `
                    <div class="output-title">💾 Статистика:</div>
                    <div class="output-text">
                        GPU: ${result.system_stats.gpu_memory_used_mb}/${result.system_stats.gpu_memory_total_mb} MB
                        RAM: ${result.system_stats.ram_used_mb}/${result.system_stats.ram_total_mb} MB
                        GPU Util: ${result.system_stats.gpu_utilization.toFixed(1)}%
                    </div>
                </div>`;
            output.innerHTML = resultHtml;
        }
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
                    )
                    result_img = images[0]
                elif mode == "img2img":
                    if not params.init_image:
                        raise HTTPException(status_code=400, detail="init_image обязателен для img2img")
                    init_img = _decode_base64_image(params.init_image)
                    strength = max(0.1, min(1.0, params.strength))
                    images = sd_flux.generate_image(
                        prompt=params.prompt,
                        negative_prompt=negative,
                        init_image=init_img,
                        strength=strength,
                        seed=seed,
                        sample_steps=params.num_inference_steps,
                        cfg_scale=params.guidance_scale,
                        width=params.width,
                        height=params.height,
                    )
                    result_img = images[0]
                elif mode == "inpaint":
                    if not params.init_image or not params.mask_image:
                        raise HTTPException(status_code=400, detail="init_image и mask_image обязательны для inpaint")
                    init_img = _decode_base64_image(params.init_image)
                    mask_img = _decode_base64_image(params.mask_image)
                    strength = max(0.1, min(1.0, params.strength))
                    images = sd_flux.generate_image(
                        prompt=params.prompt,
                        negative_prompt=negative,
                        init_image=init_img,
                        mask_image=mask_img,
                        strength=strength,
                        seed=seed,
                        sample_steps=params.num_inference_steps,
                        cfg_scale=params.guidance_scale,
                        width=params.width,
                        height=params.height,
                    )
                    result_img = images[0]
                elif mode == "upscale":
                    if not params.init_image:
                        raise HTTPException(status_code=400, detail="init_image обязателен для upscale")
                    init_img = _decode_base64_image(params.init_image)
                    factor = max(1.0, min(4.0, params.upscale_factor or 2.0))
                    if hasattr(sd_flux, "upscale"):
                        result_img = sd_flux.upscale(image=init_img, factor=factor)
                    else:
                        raise HTTPException(status_code=501, detail="upscale не поддерживается flux сборкой")
                else:
                    raise HTTPException(status_code=400, detail=f"Неизвестный режим: {mode}")

            except HTTPException:
                raise
            except Exception as e:
                print(f"❌ Flux ошибка: {e}")
                raise HTTPException(status_code=500, detail=str(e))

            img_base64 = _image_to_base64(result_img)
            stats = get_system_stats()

            return {
                "status": "✅ Успешно",
                "model": "Flux (stable_diffusion_cpp)",
                "prompt": params.prompt,
                "mode": mode,
                "image": f"data:image/png;base64,{img_base64}",
                "generation_params": params.dict(),
                "system_stats": stats.dict(),
                "timestamp": datetime.now().isoformat(),
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
                "timestamp": datetime.now().isoformat(),
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


                "model": "Realistic Vision v5.1",
async def unload_model(model_name: str):
    """Выгружает модель из памяти для освобождения GPU"""
                "mode": "text2img",
        raise HTTPException(status_code=400, detail=f"Неизвестная модель: {model_name}")

    model_manager.unload_model(model_name)
    torch.cuda.empty_cache()

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
                        <option value="turbo">Realistic Vision v5.1 (diffusers)</option>
                        <option value="qwen">Qwen3-4B (text encoder)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="prompt">Описание изображения:</label>
                    <textarea id="prompt" placeholder="Введите описание того, что вы хотите сгенерировать..."></textarea>
                </div>

                <div class="form-group">
                    <label for="negprompt">Negative prompt (опционально):</label>
                    <textarea id="negprompt" placeholder="Что нужно исключить из кадра"></textarea>
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
                                <div class="stat-label">Realistic Vision v5.1</div>
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
                    <div class="stat-label">Realistic Vision v5.1</div>
                    <div class="stat-value">${turboLoaded ? '✅ Загружена' : '❌ Не загружена'}</div>
                `;
            } catch (error) {
                console.error('Ошибка проверки моделей:', error);
            }
        }

        function onModelChange() {
            const model = document.getElementById('model').value;
            const modeGroup = document.getElementById('modeGroup');
            const strengthGroup = document.getElementById('strengthGroup');
            const upscaleGroup = document.getElementById('upscaleGroup');
            const initGroup = document.getElementById('initImageGroup');
            const maskGroup = document.getElementById('maskImageGroup');

            if (model === 'flux') {
                modeGroup.style.display = 'block';
            } else {
                modeGroup.style.display = 'none';
                document.getElementById('mode').value = 'text2img';
            }

            onModeChange();
        }

        function onModeChange() {
            const mode = document.getElementById('mode').value;
            const strengthGroup = document.getElementById('strengthGroup');
            const upscaleGroup = document.getElementById('upscaleGroup');
            const initGroup = document.getElementById('initImageGroup');
            const maskGroup = document.getElementById('maskImageGroup');

            const needsInit = mode === 'img2img' || mode === 'inpaint' || mode === 'upscale';

            strengthGroup.style.display = needsInit && mode !== 'upscale' ? 'block' : 'none';
            initGroup.style.display = needsInit ? 'block' : 'none';
            maskGroup.style.display = mode === 'inpaint' ? 'block' : 'none';
            upscaleGroup.style.display = mode === 'upscale' ? 'block' : 'none';
        }

        // Генерация изображения
        async function generateImage() {
            const prompt = document.getElementById('prompt').value;
            if (!prompt.trim()) {
                showMessage('Пожалуйста, введите описание', 'error');
                return;
            }

            const model = document.getElementById('model').value;
            const negative = document.getElementById('negprompt').value;
            
            const params = {
                prompt: prompt,
                negative_prompt: negative || null,
                model: model,
                mode: 'text2img',
                num_inference_steps: parseInt(document.getElementById('steps').value),
                guidance_scale: parseFloat(document.getElementById('guidance').value),
                temperature: parseFloat(document.getElementById('temperature').value),
                width: parseInt(document.getElementById('width').value),
                height: parseInt(document.getElementById('height').value),
                seed: document.getElementById('seed').value ? parseInt(document.getElementById('seed').value) : null,
                batch_size: 1,
                strength: null,
                upscale_factor: null,
                init_image: null,
                mask_image: null
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
        "info": "Используйте /ui для веб-интерфейса или /docs для API документации",
    }


@app.get("/ui")
async def get_ui():
    from fastapi.responses import HTMLResponse

    return HTMLResponse(content=HTML_UI)


# ==========================================
# 8. ЗАПУСК СЕРВЕРА
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 ImGen Server запускается...")
    print("=" * 60)
    print(f"📍 Веб-интерфейс: http://localhost:8001/ui")
    print(f"📚 API документация: http://localhost:8001/docs")
    print(f"🏥 Здоровье сервера: http://localhost:8001/health")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
