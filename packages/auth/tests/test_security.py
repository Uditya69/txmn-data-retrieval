import time

import jwt
import pytest

from auth.config import get_auth_settings
from auth.security import (
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)


def test_hash_password_produces_different_output_and_verifies():
    hashed = hash_password("correct-horse-battery-staple")
    assert hashed != "correct-horse-battery-staple"
    assert verify_password("correct-horse-battery-staple", hashed) is True


def test_verify_password_rejects_wrong_password():
    hashed = hash_password("correct-horse-battery-staple")
    assert verify_password("wrong-password", hashed) is False


def test_create_and_decode_access_token_roundtrips_user_id():
    settings = get_auth_settings()
    token = create_access_token("user-123", settings)
    assert decode_access_token(token, settings) == "user-123"


def test_decode_access_token_rejects_garbage_token():
    settings = get_auth_settings()
    assert decode_access_token("not-a-real-token", settings) is None


def test_decode_access_token_rejects_expired_token():
    settings = get_auth_settings()
    expired_payload = {"user_id": "user-123", "exp": int(time.time()) - 10}
    expired_token = jwt.encode(expired_payload, settings.jwt_secret, algorithm="HS256")
    assert decode_access_token(expired_token, settings) is None


def test_decode_access_token_rejects_wrong_secret():
    settings = get_auth_settings()
    token = jwt.encode(
        {"user_id": "user-123", "exp": int(time.time()) + 3600},
        "a-different-secret",
        algorithm="HS256",
    )
    assert decode_access_token(token, settings) is None


def test_generate_refresh_token_produces_distinct_unguessable_tokens():
    a = generate_refresh_token()
    b = generate_refresh_token()
    assert a != b
    assert len(a) >= 32


def test_hash_refresh_token_is_deterministic_and_one_way():
    token = generate_refresh_token()
    hashed = hash_refresh_token(token)
    assert hashed == hash_refresh_token(token)  # same input -> same hash, for lookup-by-hash
    assert hashed != token
