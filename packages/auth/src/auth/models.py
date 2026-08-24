from pydantic import BaseModel, Field


class SignupRequest(BaseModel):
    email: str
    password: str = Field(max_length=72)


class LoginRequest(BaseModel):
    email: str
    password: str = Field(max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str
