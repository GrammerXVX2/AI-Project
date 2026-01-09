import asyncio
import gc
import os
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from llama_cpp import Llama

try:
    from stable_diffusion_cpp import StableDiffusion
except ImportError:
    StableDiffusion = None

from backend.app.config import BASE_DIR, MODELS_CONFIG
from backend.app.utils.system_monitor import SystemMonitor


class HFSummarizer:
    def __init__(self, model_id: str, device_preference: str = "cuda", max_new_tokens: int = 160):
        self.model_id = model_id
        self.device = device_preference if torch.cuda.is_available() and device_preference == "cuda" else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens

    def create_chat_completion(self, messages, max_tokens=180, temperature=0.7, top_p=0.9, stop=None):
        if not messages:
            raise ValueError("No messages provided to summarizer")

        content = messages[-1].get("content", "") if isinstance(messages[-1], dict) else str(messages[-1])
        if not content:
            content = "\n".join([m.get("content", "") for m in messages if isinstance(m, dict)])

        inputs = self.tokenizer(content, return_tensors="pt", truncation=True).to(self.device)

        gen_kwargs = {
            "max_new_tokens": min(self.max_new_tokens, max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "no_repeat_ngram_size": 4,
            "do_sample": True,
        }

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        result_text = decoded
        if decoded.startswith(content):
            result_text = decoded[len(content):].strip()

        return {"choices": [{"message": {"content": result_text}}]}


class ModelManager:
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.monitor = SystemMonitor()

        for name, config in MODELS_CONFIG.items():
            self.monitor.register_model(name, None, config)
            self.monitor.set_model_status(name, "unloaded")

    def load_model(self, name: str, gpu_layers_override: Optional[int] = None) -> Any:
        if name in self.loaded_models:
            return self.loaded_models[name]

        if name not in MODELS_CONFIG:
            raise ValueError(f"Unknown model: {name}")

        config = MODELS_CONFIG[name]
        model_type = config.get("type", "chat")

        print(f"[SYSTEM] Loading {name} ({model_type})...")

        try:
            if model_type == "image":
                if StableDiffusion is None:
                    raise ImportError("stable_diffusion_cpp not installed")

                original_cwd = os.getcwd()
                try:
                    os.chdir(BASE_DIR)
                    model = StableDiffusion(
                        model_path=config["path"],
                        wtype=config.get("wtype", "default"),
                    )
                finally:
                    os.chdir(original_cwd)

            elif model_type == "hf_summary":
                model = HFSummarizer(
                    model_id=config["path"],
                    device_preference=config.get("device", "cuda"),
                    max_new_tokens=config.get("max_new_tokens", 160),
                )

            else:
                layers = config.get("n_gpu_layers", -1)
                if gpu_layers_override is not None:
                    layers = gpu_layers_override

                model = Llama(
                    model_path=config["path"],
                    n_ctx=config.get("n_ctx", 2048),
                    n_gpu_layers=layers,
                    verbose=False,
                )

            self.loaded_models[name] = model
            self.monitor.set_model_status(name, "idle")
            return model

        except Exception as e:
            print(f"[ERROR] Failed to load {name}: {e}")
            self.monitor.set_model_status(name, "error")
            return None

    def unload_model(self, name: str):
        if name in self.loaded_models:
            print(f"[SYSTEM] Unloading {name}...")
            del self.loaded_models[name]
            gc.collect()
            self.monitor.set_model_status(name, "unloaded")

    def get_model(self, name: str) -> Optional[Any]:
        return self.loaded_models.get(name)

    def set_model_active(self, name: str):
        self.monitor.set_model_active(name)

    def set_all_idle(self):
        self.monitor.set_all_idle()

    def get_monitor_stats(self):
        return self.monitor.get_stats()


model_manager = ModelManager()
