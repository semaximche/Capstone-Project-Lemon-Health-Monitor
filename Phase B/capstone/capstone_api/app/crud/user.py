from sqlalchemy.orm import Session
from app.db.models import User
from app.core.security import hash_password

class UserCRUD:

    def __init__(self):
        pass

    # Create a new user
    def create_user(self, db: Session, username: str, email: str, password: str) -> User:
        hashed_pw = hash_password(password)
        user = User(username=username, email=email, hashed_password=hashed_pw)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    # Get user by username
    def get_user_by_username(self, db: Session, user_name: str) -> User | None:
        return db.query(User).filter(User.user_name == user_name).first()

    # Get user by id
    def get_user_by_id(self, db: Session, user_id: int) -> User | None:
        return db.query(User).filter(User.id == user_id).first()

    # Update user
    def update_user(self, db: Session, user: User, **kwargs) -> User:
        for key, value in kwargs.items():
            setattr(user, key, value)
        db.commit()
        db.refresh(user)
        return user

    # Delete user
    def delete_user(self, db: Session, user: User) -> bool:
        db.delete(user)
        db.commit()
        return True


user_crud = UserCRUD()