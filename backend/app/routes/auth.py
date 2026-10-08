"""Compatibility endpoints for local/demo mode with no sign-in screen."""
from fastapi import APIRouter, Depends
from app.auth import get_current_user
from app.models import User
from app.schemas import MessageResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)) -> User:
    """Return the local demo profile for compatibility with the frontend."""
    return current_user


@router.post("/logout", response_model=MessageResponse)
async def logout() -> MessageResponse:
    """No-op: local demo mode does not have a login session."""
    return MessageResponse(message="Logout is disabled in local demo mode.")
