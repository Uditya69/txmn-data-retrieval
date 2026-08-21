import asyncio

import pytest


def test_lifespan_raises_if_classifier_artifact_missing(monkeypatch):
    import common.instant_classifier as classifier_module
    from retrieval_api.main import app

    def _raise(*args, **kwargs):
        raise FileNotFoundError("artifact missing")

    monkeypatch.setattr(classifier_module, "_load", _raise)

    async def _run():
        async with app.router.lifespan_context(app):
            pass

    with pytest.raises(FileNotFoundError):
        asyncio.run(_run())
