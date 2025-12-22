import psutil
import pynvml
import time
from typing import Dict, Any

class SystemMonitor:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        try:
            pynvml.nvmlInit()
            self.has_gpu = True
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            print(f"[MONITOR] GPU monitoring failed: {e}")
            self.has_gpu = False
            self.gpu_handle = None

    def register_model(self, name: str, model_obj: Any, config: Dict[str, Any]):
        """
        Register a model to track its status.
        config can contain: n_ctx, n_gpu_layers, path, etc.
        """
        self.models[name] = {
            "status": "idle",
            "config": config,
            "last_used": None
        }

    def set_model_active(self, name: str):
        # Only set active if it's not unloaded
        if name in self.models and self.models[name]["status"] == "unloaded":
            return

        for k in self.models:
            if self.models[k]["status"] == "generating":
                self.models[k]["status"] = "idle"
        
        if name in self.models:
            self.models[name]["status"] = "generating"
            self.models[name]["last_used"] = time.time()

    def set_model_status(self, name: str, status: str):
        if name in self.models:
            self.models[name]["status"] = status

    def set_all_idle(self):
        for k in self.models:
            if self.models[k]["status"] == "generating":
                self.models[k]["status"] = "idle"

    def get_stats(self):
        # CPU & RAM
        cpu_p = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        
        stats = {
            "cpu_percent": cpu_p,
            "ram_percent": mem.percent,
            "ram_used_gb": round(mem.used / (1024**3), 1),
            "ram_total_gb": round(mem.total / (1024**3), 1),
            "gpu_stats": None,
            "models": {}
        }

        # GPU
        if self.has_gpu and self.gpu_handle:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                name = pynvml.nvmlDeviceGetName(self.gpu_handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                stats["gpu_stats"] = {
                    "name": name,
                    "load": util.gpu,
                    "memory_percent": round((mem_info.used / mem_info.total) * 100, 1),
                    "memory_used_gb": round(mem_info.used / (1024**3), 1),
                    "memory_total_gb": round(mem_info.total / (1024**3), 1),
                    "temp": temp
                }
            except Exception as e:
                stats["gpu_error"] = str(e)

        # Models Info
        for name, data in self.models.items():
            stats["models"][name] = {
                "status": data["status"],
                "config": data["config"],
                "last_used": data["last_used"]
            }

        return stats
