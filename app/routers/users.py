from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import schemas
from app.database import get_db
from app.models import User, UserRole
from app.security import require_lecturer

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/students", response_model=List[schemas.UserResponse])
def list_students(db: Session = Depends(get_db), _: User = Depends(require_lecturer)):
    """Lecturers only (the original exposed the student list to every logged-in user)."""
    return (db.query(User)
            .filter(User.role == UserRole.STUDENT, User.is_active.is_(True))
            .order_by(User.full_name)
            .all())
