# app/db/init_db.py
from sqlalchemy.orm import Session
from app.db.db import Base, engine
from app.db.models import User
from werkzeug.security import generate_password_hash
import uuid

def init_db():

    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")

    initial_users = [
        {"user_name": "david", "password": "david"},
        {"user_name": "maxim", "password": "maxim"},
    ]

    # Use session to add users
    with Session(engine) as session:
        for u in initial_users:
            user = User(user_name=u["user_name"], password=u["password"])
            session.add(user)
        session.commit()

    print("Initial users added!")


if __name__ == "__main__":
    init_db()
