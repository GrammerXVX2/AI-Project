import asyncio
import gc
import os
from typing import Any, Dict, Optional

from llama_cpp import Llama

try:
    from stable_diffusion_cpp import StableDiffusion
except ImportError:
    StableDiffusion = None

from ai_server.app.config import BASE_DIR, MODELS_CONFIG
from ai_server.app.utils.system_monitor import SystemMonitor


class ModelManager:
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.monitor = SystemMonitor()

        # Register all known models in monitor as unloaded
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

                # Fix for Cyrillic paths/Windows: Change CWD to project root temporarily
                # and use relative path defined in config
                original_cwd = os.getcwd()
                try:
                    os.chdir(BASE_DIR)
                    model = StableDiffusion(
                        model_path=config["path"],
                        wtype=config.get("wtype", "default"),  # default, f16, q8_0 etc if supported
                    )
                finally:
                    os.chdir(original_cwd)

            else:
                # LLM Loading
                layers = config["n_gpu_layers"]
                if gpu_layers_override is not None:
                    layers = gpu_layers_override

                model = Llama(
                    model_path=config["path"],
                    n_ctx=config["n_ctx"],
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


# Global instance
model_manager = ModelManager()
