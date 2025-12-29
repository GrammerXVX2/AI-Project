from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
import uuid

from ai_server.app.services.image_generation_service import image_service
from ai_server.app.services.model_manager import model_manager
from ai_server.app.services.history_manager import history_manager

router = APIRouter(prefix="/api/image", tags=["image"])

class ImageRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 9
    seed: int = 42
    enhance_prompt: bool = True
    low_vram: bool = False
    offload_cpu: bool = True

@router.post("/generate")
async def generate_image(request: ImageRequest):
    final_prompt = request.prompt
    
    # 1. Session Management
    current_session_id = request.session_id or str(uuid.uuid4())
    
    # Load history to append messages later
    history = await history_manager.aload_history(current_session_id)

    if request.enhance_prompt:
        orchestrator = model_manager.get_model("Orchestrator")

        if orchestrator:
            print("[ImageAPI] Enhancing prompt with Orchestrator...")
            # Simple prompt engineering for the orchestrator
            system_msg = "You are an expert prompt engineer for Stable Diffusion. Convert the user's request into a detailed, high-quality English prompt for image generation. Focus on visual details, lighting, style, and composition. Output ONLY the prompt, nothing else."
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": request.prompt}
            ]
            
            try:
                # Run in thread to not block
                response = await asyncio.to_thread(
                    orchestrator.create_chat_completion,
                    messages=messages,
                    max_tokens=256,
                    temperature=0.7
                )
                enhanced = response['choices'][0]['message']['content'].strip()
                print(f"[ImageAPI] Enhanced prompt: {enhanced}")
                final_prompt = enhanced
            except Exception as e:
                print(f"[ImageAPI] Failed to enhance prompt: {e}")
        else:
            print("[ImageAPI] Orchestrator not loaded, using original prompt.")

    try:
        # Run generation in thread
        gen_result = await asyncio.to_thread(
            image_service.generate,
            prompt=final_prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            seed=request.seed,
            low_vram=request.low_vram,
            offload_cpu=request.offload_cpu,
            session_id=current_session_id,
        )

        if isinstance(gen_result, dict):
            file_url = gen_result.get("file_url")
            duration = gen_result.get("duration")
            if file_url and file_url.startswith("/"):
                # Serve absolute so frontend (different port) can load it
                image_url = f"http://localhost:8000{file_url}"
            else:
                image_url = file_url or gen_result.get("data_url")
            data_url = gen_result.get("data_url")
        else:
            image_url = gen_result
            data_url = gen_result
            file_url = None
            duration = None
        
        # Save to history
        user_msg = {"role": "user", "content": request.prompt}
        assistant_msg = {
            "role": "assistant", 
            "content": f"![Generated Image]({image_url})\n\n*Prompt: \"{final_prompt}\"*",
            "meta": {
                "model": "Z-Image-Turbo",
                "task": "image_generation",
                "image_url": image_url,
                "file_url": file_url,
                "duration": duration
            }
        }
        
        history.append(user_msg)
        history.append(assistant_msg)
        await history_manager.asave_history(current_session_id, history)
        
        return {
            "status": "success",
            "image": image_url,
            "image_file": file_url,
            "image_data_url": data_url,
            "prompt_used": final_prompt,
            "session_id": current_session_id,
            "duration": duration
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{session_id}")
async def get_progress(session_id: str):
    progress = image_service.get_progress(session_id)
    if not progress:
        return {"status": "idle"}
    return {"status": "running", **progress}
