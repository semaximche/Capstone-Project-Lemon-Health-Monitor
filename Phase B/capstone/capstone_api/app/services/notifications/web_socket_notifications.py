from fastapi import WebSocket

active_connections = {}

async def connect(user_id: str, websocket: WebSocket):
    await websocket.accept()
    active_connections[user_id] = websocket
    print("new connection added:", user_id)

async def disconnect(user_id: str):
    active_connections.pop(user_id, None)

async def notify_user(user_id: str, message):
    ws = active_connections.get(user_id)
    if ws:
        await ws.send_json(message)
