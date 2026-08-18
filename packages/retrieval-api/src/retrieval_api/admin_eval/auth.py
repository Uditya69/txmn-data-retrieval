import common.config


def is_valid_admin_token(token: str | None) -> bool:
    """True only when ADMIN_SECRET is set AND token matches it exactly. A pure
    predicate (not exception-raising) so both the WS route (needs a custom close
    code, not an HTTPException) and the REST cache-read endpoint (needs an
    HTTPException) can each decide their own rejection shape."""
    settings = common.config.get_settings()
    return bool(settings.admin_secret) and token == settings.admin_secret
