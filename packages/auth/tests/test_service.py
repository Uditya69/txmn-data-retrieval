import pytest

from auth.security import hash_password
from auth.service import (
    EmailAlreadyRegistered,
    InvalidCredentials,
    login,
    signup,
)


@pytest.mark.asyncio
async def test_signup_creates_user_and_returns_user_id(fake_users_collection):
    users = fake_users_collection
    user_id = await signup(users, "alice@example.com", "hunter2")
    assert isinstance(user_id, str) and user_id
    stored = users.documents[0]
    assert stored["email"] == "alice@example.com"
    assert stored["_id"] == user_id
    assert stored["password_hash"] != "hunter2"


@pytest.mark.asyncio
async def test_signup_rejects_duplicate_email(fake_users_collection):
    users = fake_users_collection
    await signup(users, "alice@example.com", "hunter2")
    with pytest.raises(EmailAlreadyRegistered):
        await signup(users, "alice@example.com", "different-password")


@pytest.mark.asyncio
async def test_login_returns_user_id_for_correct_credentials(fake_users_collection):
    users = fake_users_collection
    user_id = await signup(users, "alice@example.com", "hunter2")
    result = await login(users, "alice@example.com", "hunter2")
    assert result == user_id


@pytest.mark.asyncio
async def test_login_rejects_unknown_email(fake_users_collection):
    users = fake_users_collection
    with pytest.raises(InvalidCredentials):
        await login(users, "nobody@example.com", "hunter2")


@pytest.mark.asyncio
async def test_login_rejects_wrong_password(fake_users_collection):
    users = fake_users_collection
    await signup(users, "alice@example.com", "hunter2")
    with pytest.raises(InvalidCredentials):
        await login(users, "alice@example.com", "wrong-password")
