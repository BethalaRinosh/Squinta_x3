"""Local demo-user identity for Squinta.

Google sign-in is disabled in local/demo mode. Existing API routes still
receive a real internal User row, preserving user_id-based document/model
ownership and storage paths. Do not expose this no-login mode on a public
multi-user deployment.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.database import get_db
from app.models import User

bearer_scheme = HTTPBearer(auto_error=False)
DEMO_GOOGLE_ID = "squinta-local-demo"
DEMO_EMAIL = "demo@squinta.local"
DEMO_NAME = "Squinta Demo User"


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Return the stable local demo user without requiring a login token."""
    result = await db.execute(select(User).where(User.google_id == DEMO_GOOGLE_ID))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(
            google_id=DEMO_GOOGLE_ID,
            email=DEMO_EMAIL,
            name=DEMO_NAME,
            picture_url=None,
            google_access_token=None,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user
