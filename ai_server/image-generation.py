import argparse
import os
from typing import Any

import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from diffusers.quantizers.quantization_config import GGUFQuantizationConfig
from diffusers.models.attention_processor import AttnProcessor
from huggingface_hub import hf_hub_download
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", action="store_true", help="Перекачать модель заново")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Куда грузить (cuda/auto/cpu)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--low_vram", action="store_true", help="Экономить VRAM (attention slicing/offload/снижение разрешения)")
    parser.add_argument("--steps", type=int, default=9, help="Количество шагов (Turbo: 8–10, меньше экономит память)")
    args = parser.parse_args()

    # Определяем девайс с возможностью падения на CPU
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA недоступна, переключаю на CPU")
        device = "cpu"
    elif args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Держим bfloat16 на CUDA (как рекомендует Turbo), CPU — float32
    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    prompt = (
        "Photorealistic style: an elegant Japanese woman wearing a refined fabric face mask, a fashionable cropped top, and a short pleated skirt, standing confidently against an Osaka cityscape. In the background, cherry blossom trees are in full bloom, with soft pink petals carried gently by a light spring breeze. The background is subtly blurred—either featuring Osaka Castle or a neon-lit urban nightscape, depending on the time of day. Soft daylight or warm evening lighting accentuates natural skin textures, fabric details, and the early spring atmosphere. The color palette is restrained, with delicate pink and pastel tones. Highly detailed, sharp focus on the woman’s face and figure, with a bokeh effect in the background for depth."
    )

    # Качаем GGUF корректно через hf_hub_download, а не через blob-ссылку
    gguf_path = hf_hub_download(
        repo_id="jayn7/Z-Image-Turbo-GGUF",
        filename="z_image_turbo-Q3_K_S.gguf",
        force_download=args.force_download,
        resume_download=not args.force_download,
    )

    transformer = ZImageTransformer2DModel.from_single_file(
        gguf_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
        dtype=compute_dtype,
    )

    pipeline = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        transformer=transformer,
        torch_dtype=compute_dtype,
    )

    # Уходим от SDPA с enable_gqa (несовместимо с текущей версией torch) на простой процессор внимания
    # В этой версии ZImagePipeline нет set_attn_processor; применяем к transformer напрямую
    if hasattr(pipeline, "transformer") and hasattr(pipeline.transformer, "set_attn_processor"):
        pipeline.transformer.set_attn_processor(AttnProcessor())

    # Low-VRAM tweaks
    if args.low_vram:
        # Снизим запрошенную резолюцию, если она слишком большая для 4GB
        args.height = min(args.height, 2048)
        args.width = min(args.width, 2048)
        pipeline.enable_attention_slicing()

    # Offload strategy
    if device == "cuda":
        if args.low_vram:
            # Не грузим всю модель в GPU: оставляем на CPU и используем model_cpu_offload с выгрузкой слоёв
            pipeline.enable_model_cpu_offload()
            generator_device = "cuda"
        else:
            pipeline = pipeline.to("cuda")
            pipeline.enable_model_cpu_offload()
            generator_device = "cuda"
    else:
        pipeline = pipeline.to("cpu")
        pipeline.enable_model_cpu_offload()
        generator_device = "cpu"

    generator = torch.Generator(generator_device).manual_seed(args.seed)

    try:
        result = pipeline(
            prompt=prompt,
            num_inference_steps=args.steps,  # Turbo советует 8–10
            guidance_scale=0.0,     # Turbo — без гида
            height=args.height,
            width=args.width,
            generator=generator,
        )
    except torch.cuda.OutOfMemoryError:
        print("⚠️ CUDA OOM, переключаю на CPU")
        torch.cuda.empty_cache()
        pipeline = pipeline.to("cpu")
        pipeline.enable_model_cpu_offload()
        generator = torch.Generator("cpu").manual_seed(args.seed)
        result = pipeline(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            height=args.height,
            width=args.width,
            generator=generator,
        )

    image: Any = result.images[0] if hasattr(result, "images") else result[0]  # type: ignore[attr-defined]
    if isinstance(image, torch.Tensor):
        try:
            from torchvision.transforms.functional import to_pil_image

            image = to_pil_image(image)
        except Exception:
            # Если нет torchvision, сохраняем через PIL из numpy
            image = Image.fromarray(image.cpu().numpy()) if hasattr(image, "cpu") else Image.fromarray(image)  # type: ignore[arg-type]

    if hasattr(image, "save"):
        image.save("zimage.png")
    else:
        raise RuntimeError("Не удалось получить PIL-совместимое изображение из вывода пайплайна")
    print("✅ Сохранено: zimage.png")


if __name__ == "__main__":
    main()
