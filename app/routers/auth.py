from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app import schemas
from app.config import settings
from app.database import get_db
from app.models import User, UserRole
from app.security import (create_access_token, get_current_user, hash_password, verify_password)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Pre-computed so that login takes the same time whether or not the email exists.
_DUMMY_HASH = hash_password("dummy-password-1")


def _authenticate(db: Session, email: str, password: str) -> User:
    user = db.query(User).filter(User.email == email.strip().lower()).first()
    ok = verify_password(password, user.hashed_password if user else _DUMMY_HASH)
    if not user or not ok or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    return user


@router.post("/register", response_model=schemas.UserResponse, status_code=201)
def register(data: schemas.UserRegister, db: Session = Depends(get_db)):
    if data.role == UserRole.LECTURER and settings.LECTURER_INVITE_CODE:
        if not hmac.compare_digest(data.invite_code or "", settings.LECTURER_INVITE_CODE):
            raise HTTPException(status_code=403, detail="A valid lecturer invite code is required")
    email = data.email.strip().lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(full_name=data.full_name, email=email,
                hashed_password=hash_password(data.password), role=data.role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=schemas.Token)
def login(data: schemas.UserLogin, db: Session = Depends(get_db)):
    user = _authenticate(db, data.email, data.password)
    return {"access_token": create_access_token(user), "token_type": "bearer", "user": user}


@router.post("/token", response_model=schemas.Token, include_in_schema=True)
def login_form(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Form-encoded login used by Swagger UI's Authorize button (username = email)."""
    user = _authenticate(db, form.username, form.password)
    return {"access_token": create_access_token(user), "token_type": "bearer", "user": user}


@router.get("/me", response_model=schemas.UserResponse)
def me(user: User = Depends(get_current_user)):
    return user
