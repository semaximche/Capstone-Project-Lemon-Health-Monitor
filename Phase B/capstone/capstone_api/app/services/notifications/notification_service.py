from app.services.notifications.web_socket_notifications import notify_user

async def handle_notification(event):
    notification = {
        "user_id": event["user_id"],
        "analysis_id": event["analysis_id"],
    }

    # await save_notification(notification)  TODO: can save the notifications later if needed
    print("notifying user with notification -  analysis id:" + notification["analysis_id"])
    await notify_user(event["user_id"], notification)
