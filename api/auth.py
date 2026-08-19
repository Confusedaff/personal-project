"""
Authentication router and dependencies.

Provides JWT-based authentication with per-user database isolation.
"""
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel

from api.user_db import (
    get_user_by_id,
    get_user_by_email,
    get_user_by_username,
    create_user,
    init_users_db,
)
from api.schemas import UserCreate, Token, UserResponse

# ── Configuration ─────────────────────────────────────────────────────

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("TOKEN_EXPIRE_MINUTES", "1440"))  # 24h

# ── OAuth2 scheme ─────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

# ── Router ────────────────────────────────────────────────────────────

router = APIRouter()

# Initialize the shared users database on import
init_users_db()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def hash_password(password: str) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> dict:
    """Dependency: validates the JWT and returns the user dict."""
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = get_user_by_id(int(user_id))
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account disabled")
    return user


# ── Auth endpoints ────────────────────────────────────────────────────

@router.post("/register", response_model=UserResponse, status_code=201)
async def register(req: UserCreate):
    """Register a new user. Creates a private data directory for them."""
    if get_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    if get_user_by_username(req.username):
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed = hash_password(req.password)
    user_id = create_user(req.username, req.email, hashed)
    return UserResponse(id=user_id, username=req.username, email=req.email, is_active=True)


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login with email + password. Returns a JWT."""
    user = get_user_by_email(form_data.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account disabled")

    token = create_access_token(data={"sub": str(user["id"]), "username": user["username"]})
    return Token(
        access_token=token,
        user_id=user["id"],
        username=user["username"],
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's info."""
    return UserResponse(
        id=user["id"],
        username=user["username"],
        email=user["email"],
        is_active=bool(user.get("is_active", True)),
    )


@router.post("/logout")
async def logout():
    """Client-side logout (JWT is stateless — just discard the token)."""
    return {"detail": "Logged out. Discard the token on the client."}
