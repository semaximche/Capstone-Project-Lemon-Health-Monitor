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
            # Check if user already exists
            existing_user = session.query(User).filter(User.user_name == u["user_name"]).first()
            if not existing_user:
                user = User(user_name=u["user_name"], password=u["password"])
                session.add(user)
                print(f"Added user: {u['user_name']}")
            else:
                print(f"User {u['user_name']} already exists, skipping.")
        session.commit()

    print("Database initialization complete!")


if __name__ == "__main__":
    init_db()
