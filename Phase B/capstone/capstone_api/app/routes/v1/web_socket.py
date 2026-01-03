from fastapi import WebSocket, APIRouter
from app.services.notifications.web_socket_notifications import connect, disconnect

router = APIRouter()

@router.websocket("/ws/notifications/{user_id}")
async def notifications_ws(websocket: WebSocket, user_id: str):
    await connect(user_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        await disconnect(user_id)
