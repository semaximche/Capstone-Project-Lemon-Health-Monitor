from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
import hashlib
from app.settings import settings

# bcrypt context
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# def hash_password(password: str) -> str:
#     """
#     Hash a password safely, even if it's longer than 72 bytes.
#     Uses SHA256 pre-hash + bcrypt.
#     """
#     # Pre-hash password using SHA256
#     sha256_pass = hashlib.sha256(password.encode()).digest()
#     # Then hash with bcrypt
#     return pwd_context.hash(sha256_pass)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against the hashed password.
    """
    return plain_password == hashed_password
    # sha256_pass = hashlib.sha256(plain_password.encode()).digest()
    # return pwd_context.verify(sha256_pass, hashed_password)

def create_access_token(data: dict, expires_minutes: int) -> str:
    """
    Create a JWT token with an expiration time.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    
    # Read JWT secret key from settings (which loads from .env file)
    if not settings.jwt_secret_key:
        raise RuntimeError(
            "JWT_SECRET_KEY is not set. Please set it in your .env file."
        )

    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
