from fastapi import APIRouter, HTTPException

from ai_server.app.services.model_manager import model_manager

router = APIRouter(prefix="/api/models", tags=["models"])


@router.post("/{name}/load")
async def load_model(name: str, mode: str = "gpu"):
    try:
        # Logic for mode -> layers mapping
        layers = -1  # Default GPU
        if mode == "cpu":
            layers = 0
        elif mode == "hybrid":
            layers = 14

        model = model_manager.load_model(name, gpu_layers_override=layers)
        if not model:
            raise HTTPException(500, f"Failed to load model {name}")

        return {"status": "loaded", "name": name, "mode": mode}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/{name}/unload")
async def unload_model(name: str):
    if name == "Orchestrator":
        raise HTTPException(400, "Cannot unload orchestrator")

    model_manager.unload_model(name)
    return {"status": "unloaded", "name": name}
