from fastapi import APIRouter, Depends, HTTPException

from auth.config import get_auth_settings
from auth.db import get_mongo_client, get_users_collection
from auth.models import LoginRequest, SignupRequest, TokenResponse
from auth.security import create_access_token
from auth.service import EmailAlreadyRegistered, InvalidCredentials, login, signup

router = APIRouter(prefix="/auth", tags=["auth"])


def get_users_dependency():
    settings = get_auth_settings()
    client = get_mongo_client(settings)
    return get_users_collection(client, settings)


@router.post("/signup", response_model=TokenResponse)
async def signup_route(payload: SignupRequest, users=Depends(get_users_dependency)):
    settings = get_auth_settings()
    try:
        user_id = await signup(users, payload.email, payload.password)
    except EmailAlreadyRegistered:
        raise HTTPException(status_code=409, detail="email already registered")
    token = create_access_token(user_id, settings)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
async def login_route(payload: LoginRequest, users=Depends(get_users_dependency)):
    settings = get_auth_settings()
    try:
        user_id = await login(users, payload.email, payload.password)
    except InvalidCredentials:
        raise HTTPException(status_code=401, detail="invalid email or password")
    token = create_access_token(user_id, settings)
    return TokenResponse(access_token=token)
