import pytest

from retrieval_api.admin_eval.auth import is_valid_admin_token


def test_rejects_when_admin_secret_unset(monkeypatch):
    import common.config as config_module
    monkeypatch.setattr(config_module, "get_settings", lambda: type("S", (), {"admin_secret": None})())
    assert is_valid_admin_token("anything") is False


def test_rejects_wrong_token(monkeypatch):
    import common.config as config_module
    monkeypatch.setattr(config_module, "get_settings", lambda: type("S", (), {"admin_secret": "correct-secret"})())
    assert is_valid_admin_token("wrong-secret") is False


def test_accepts_matching_token(monkeypatch):
    import common.config as config_module
    monkeypatch.setattr(config_module, "get_settings", lambda: type("S", (), {"admin_secret": "correct-secret"})())
    assert is_valid_admin_token("correct-secret") is True


def test_rejects_none_token(monkeypatch):
    import common.config as config_module
    monkeypatch.setattr(config_module, "get_settings", lambda: type("S", (), {"admin_secret": "correct-secret"})())
    assert is_valid_admin_token(None) is False
