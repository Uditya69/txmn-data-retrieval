from fastapi import APIRouter, Depends, HTTPException, Response

from auth.config import get_auth_settings
from auth.db import get_mongo_client, get_refresh_tokens_collection, get_users_collection
from auth.models import LoginRequest, RefreshRequest, SignupRequest, TokenResponse
from auth.security import create_access_token
from auth.service import (
    EmailAlreadyRegistered,
    InvalidCredentials,
    InvalidRefreshToken,
    create_refresh_token,
    login,
    revoke_refresh_token,
    rotate_refresh_token,
    signup,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def get_users_dependency():
    settings = get_auth_settings()
    client = get_mongo_client(settings)
    return get_users_collection(client, settings)


def get_refresh_tokens_dependency():
    settings = get_auth_settings()
    client = get_mongo_client(settings)
    return get_refresh_tokens_collection(client, settings)


@router.post("/signup", response_model=TokenResponse)
async def signup_route(
    payload: SignupRequest, users=Depends(get_users_dependency), refresh_tokens=Depends(get_refresh_tokens_dependency),
):
    settings = get_auth_settings()
    try:
        user_id = await signup(users, payload.email, payload.password)
    except EmailAlreadyRegistered:
        raise HTTPException(status_code=409, detail="email already registered")
    access_token = create_access_token(user_id, settings)
    refresh_token = await create_refresh_token(refresh_tokens, user_id, settings)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/login", response_model=TokenResponse)
async def login_route(
    payload: LoginRequest, users=Depends(get_users_dependency), refresh_tokens=Depends(get_refresh_tokens_dependency),
):
    settings = get_auth_settings()
    try:
        user_id = await login(users, payload.email, payload.password)
    except InvalidCredentials:
        raise HTTPException(status_code=401, detail="invalid email or password")
    access_token = create_access_token(user_id, settings)
    refresh_token = await create_refresh_token(refresh_tokens, user_id, settings)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_route(payload: RefreshRequest, refresh_tokens=Depends(get_refresh_tokens_dependency)):
    settings = get_auth_settings()
    try:
        user_id, new_refresh_token = await rotate_refresh_token(refresh_tokens, payload.refresh_token, settings)
    except InvalidRefreshToken:
        raise HTTPException(status_code=401, detail="refresh token invalid or expired")
    access_token = create_access_token(user_id, settings)
    return TokenResponse(access_token=access_token, refresh_token=new_refresh_token)


@router.post("/logout", status_code=204)
async def logout_route(payload: RefreshRequest, refresh_tokens=Depends(get_refresh_tokens_dependency)):
    # Best-effort revocation, not a security boundary on its own (the access
    # token already issued stays valid until its own short expiry regardless -
    # this only prevents the refresh token from minting further access tokens
    # after this point). No error on an already-invalid/unknown token; logout
    # should always succeed from the client's point of view.
    await revoke_refresh_token(refresh_tokens, payload.refresh_token)
    return Response(status_code=204)
