import asyncio

from fastapi import APIRouter, WebSocket

from ai_server.app.services.model_manager import model_manager

router = APIRouter(tags=["system"])


@router.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Get stats from monitor via model_manager
            data = await asyncio.to_thread(model_manager.get_monitor_stats)
            await websocket.send_json(data)
            await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket disconnected: {e}")
