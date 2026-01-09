from fastapi import WebSocket

active_connections = {}

async def connect(user_id: str, websocket: WebSocket):
    # Check if user already has an active connection
    existing_connection = active_connections.get(user_id)
    if not existing_connection:
        # Close the existing connection
        await websocket.accept()
        active_connections[user_id] = websocket
        print("new connection added:", user_id)
    else:
        print("user already connected:", user_id)


async def disconnect(user_id: str):
    connection = active_connections.pop(user_id, None)
    if connection:
        try:
            await connection.close()
            print(f"Disconnected user: {user_id}")
        except Exception as e:
            print(f"Error closing connection for user {user_id}: {e}")

async def notify_user(user_id: str, message):
    ws = active_connections.get(user_id)
    if ws:
        await ws.send_json(message)
