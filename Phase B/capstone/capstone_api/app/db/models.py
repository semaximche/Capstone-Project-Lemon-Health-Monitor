from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from app.db.db import Base


class User(Base):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(primary_key=True)
    user_name: Mapped[str] = mapped_column(String(50),primary_key=True)
    password: Mapped[str] = mapped_column(String(100), unique=True)


class Product(Base):
    __tablename__ = 'products'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    price: Mapped[float] = mapped_column()