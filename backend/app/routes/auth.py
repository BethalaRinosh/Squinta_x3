"""Authentication routes -- Google OAuth2 login / callback / me / logout."""

import logging
from uuid import UUID
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

from app.auth import (
    create_access_token,
    get_current_user,
    get_or_create_user,
    oauth,
)
from app.config import settings
from app.database import get_db
from app.models import User
from app.schemas import MessageResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


class GuestSessionRequest(BaseModel):
    guest_id: str = Field(..., min_length=36, max_length=36)


@router.post("/guest")
async def create_guest_session(
    body: GuestSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Create or resume a browser-scoped guest account.

    The guest UUID is generated and persisted by the frontend. It lets ordinary
    OCR features use the same database ownership checks as Google accounts,
    without requiring OAuth or exposing other users' documents.
    """
    try:
        guest_id = str(UUID(body.guest_id))
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid guest session ID") from exc

    google_id = f"guest:{guest_id}"
    email = f"guest-{guest_id}@guest.squinta.local"
    result = await db.execute(select(User).where(User.google_id == google_id))
    user = result.scalar_one_or_none()

    if user is None:
        user = User(google_id=google_id, email=email, name="Guest")
        db.add(user)
        try:
            await db.commit()
        except IntegrityError:
            # Two tabs can initialize the same guest at once. If the other
            # request won the unique-key race, reuse that row.
            await db.rollback()
            result = await db.execute(select(User).where(User.google_id == google_id))
            user = result.scalar_one_or_none()
            if user is None:
                raise
    # Flush is unnecessary for an existing row, but commit here is harmless
    # and ensures any incidental session state is closed before returning.
    else:
        await db.commit()

    token = create_access_token(data={"sub": str(user.id)})
    return {
        "token": token,
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "is_guest": True,
        },
    }


@router.get("/login")
async def login(request: Request) -> RedirectResponse:
    """Redirect the browser to Google's OAuth2 consent screen."""
    # Use the backend's canonical callback URL instead of the frontend's proxy URL.
    # When this app is reached through Vite, request.url_for() resolves to the
    # browser-facing origin (e.g. localhost:5176), which does not match the
    # configured Google redirect URI.
    redirect_uri = f"{settings.BACKEND_URL.rstrip('/')}/auth/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback", name="auth_callback")
async def callback(request: Request, db: AsyncSession = Depends(get_db)) -> RedirectResponse:
    """Handle the OAuth2 callback from Google.

    Exchanges the authorisation code for tokens, upserts the user row,
    mints a JWT, and redirects back to the frontend with the token as a
    query parameter so the SPA can store it.
    """
    try:
        token_data = await oauth.google.authorize_access_token(request)
    except Exception:
        logger.exception("authorize_access_token failed")
        # OAuth is optional now: return to the dashboard rather than a removed
        # login page. The user can keep using guest mode if connection fails.
        return RedirectResponse(url=f"{settings.FRONTEND_URL}/?google_auth_error=1")

    # The id_token is already verified by authlib; extract user info.
    userinfo: dict = token_data.get("userinfo", {})
    if not userinfo:
        userinfo = await oauth.google.userinfo(token=token_data)

    google_id: str = userinfo["sub"]
    email: str = userinfo["email"]
    name: str = userinfo.get("name", email)
    picture: str | None = userinfo.get("picture")

    user = await get_or_create_user(db, google_id=google_id, email=email, name=name, picture_url=picture)

    # Store the Google access token in the DB so we can call
    # Google Photos API on behalf of the user later.
    user.google_access_token = token_data.get("access_token")
    await db.commit()

    jwt_token = create_access_token(data={"sub": str(user.id)})

    # Redirect to frontend with the JWT in query params.
    params = urlencode({"token": jwt_token})
    return RedirectResponse(url=f"{settings.FRONTEND_URL}/auth/callback?{params}")


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)) -> User:
    """Return the currently authenticated user's profile."""
    return current_user


@router.post("/logout", response_model=MessageResponse)
async def logout(
    request: Request,
    _current_user: User = Depends(get_current_user),
) -> MessageResponse:
    """Log out the current user.

    Since JWTs are stateless the client is responsible for discarding its
    token.  This endpoint clears any server-side session data (e.g. the
    cached Google access token) and returns a confirmation message.
    """
    request.session.clear()
    return MessageResponse(message="Logged out successfully")
