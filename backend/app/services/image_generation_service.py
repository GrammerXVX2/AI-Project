import os
import torch
import base64
import io
import time
from pathlib import Path
from typing import Optional, Any
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from diffusers.quantizers.quantization_config import GGUFQuantizationConfig
from diffusers.models.attention_processor import AttnProcessor
from huggingface_hub import hf_hub_download
from PIL import Image

from backend.app.config import STATIC_DIR

class ImageGenerationService:
    def __init__(self):
        self.pipeline: Optional[ZImagePipeline] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.current_offload_cpu = None
        self._progress = {}

    def load_model(self, offload_cpu: bool = True):
        # If pipeline exists and offload setting matches, do nothing
        if self.pipeline is not None and self.current_offload_cpu == offload_cpu:
            return
        
        # If pipeline exists but setting changed, we might need to reload or adjust.
        # For simplicity and stability, if we need to change offload strategy, let's reload.
        if self.pipeline is not None:
            print("[ImageService] Offload setting changed. Reloading model...")
            del self.pipeline
            torch.cuda.empty_cache()
            self.pipeline = None

        print(f"[ImageService] Loading Z-Image model (Offload CPU: {offload_cpu})...")
        
        # Download/Load GGUF
        gguf_path = hf_hub_download(
            repo_id="jayn7/Z-Image-Turbo-GGUF",
            filename="z_image_turbo-Q3_K_S.gguf",
        )

        transformer = ZImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=self.compute_dtype),
            dtype=self.compute_dtype,
        )

        self.pipeline = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            transformer=transformer,
            torch_dtype=self.compute_dtype,
        )

        if hasattr(self.pipeline, "transformer") and hasattr(self.pipeline.transformer, "set_attn_processor"):
            self.pipeline.transformer.set_attn_processor(AttnProcessor())

        # Offload strategy
        if self.device == "cuda":
            if offload_cpu:
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to("cuda")
        else:
            self.pipeline = self.pipeline.to("cpu")
            self.pipeline.enable_model_cpu_offload()
            
        self.current_offload_cpu = offload_cpu

    def _set_progress(self, session_id: Optional[str], step: int, total: int):
        if session_id:
            self._progress[session_id] = {"step": step, "total": total}

    def get_progress(self, session_id: Optional[str]):
        return self._progress.get(session_id)

    def clear_progress(self, session_id: Optional[str]):
        if session_id and session_id in self._progress:
            del self._progress[session_id]

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 9,
        seed: int = 42,
        low_vram: bool = False,
        offload_cpu: bool = True,
        session_id: Optional[str] = None,
    ) -> dict:
        self.load_model(offload_cpu=offload_cpu)

        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded")

        pipe: ZImagePipeline = self.pipeline

        generator = torch.Generator(self.device).manual_seed(seed)
        
        # Low VRAM adjustments
        if low_vram and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        
        try:
            # Reset progress (coarse: start then mark done at the end)
            self._set_progress(session_id, 0, steps)

            start_ts = time.time()

            result: Any = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=0.0,
                height=height,
                width=width,
                generator=generator,
            )
            if hasattr(result, "images"):
                image = result.images[0]
            else:
                # Fallback for tuple outputs
                image = result[0]
            duration = time.time() - start_ts
            
            # Save to disk for safety
            output_dir = STATIC_DIR / "generated_images"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            filename = f"img_{timestamp}_{seed}.png"
            file_path = output_dir / filename
            image.save(file_path)
            print(f"[ImageService] Saved image to {file_path}")

            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            data_url = f"data:image/png;base64,{img_str}"

            file_url = f"/static/generated_images/{filename}"

            return {
                "data_url": data_url,
                "file_url": file_url,
                "file_path": str(file_path),
                "duration": duration
            }
            
        except Exception as e:
            print(f"[ImageService] Error: {e}")
            raise e
        finally:
            # Mark as done for clients polling progress
            self._set_progress(session_id, steps, steps)
            self.clear_progress(session_id)

image_service = ImageGenerationService()
