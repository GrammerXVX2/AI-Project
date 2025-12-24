import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_server.app.api import chat, models, system
from ai_server.app.services.model_manager import model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[SYSTEM] Starting up...")
    # Load Orchestrator immediately
    try:
        await asyncio.to_thread(model_manager.load_model, "Orchestrator")
    except Exception as e:
        print(f"[ERROR] Failed to load orchestrator on startup: {e}")

    yield

    # Shutdown
    print("[SYSTEM] Shutting down...")
    # Clean up if needed


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(system.router)


@app.get("/")
async def root():
    return {"status": "AI Server Running", "version": "2.0.0"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
