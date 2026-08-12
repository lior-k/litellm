"""Shared fixtures for tests in claude-code-packaging/tests.

Stubs out ``wonderfence_sdk`` (not pip-installed in dev) and adds the
packaging directory to ``sys.path`` so ``wonderfence_guardrail`` imports.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

PKG_DIR = Path(__file__).resolve().parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))


def _install_sdk_stub(client_factory=None):
    sdk = Mock()
    client_pkg = Mock()
    models_pkg = Mock()

    factory = client_factory or (lambda **kwargs: Mock(close=AsyncMock()))
    client_pkg.WonderFenceV2Client = Mock(side_effect=factory)
    sdk.client = client_pkg
    models_pkg.AnalysisContext = Mock(return_value=Mock())
    sdk.models = models_pkg
    sys.modules["wonderfence_sdk"] = sdk
    sys.modules["wonderfence_sdk.client"] = client_pkg
    sys.modules["wonderfence_sdk.models"] = models_pkg


_install_sdk_stub()


@pytest.fixture
def mock_client():
    c = Mock()
    c.evaluate_prompt = AsyncMock()
    c.evaluate_response = AsyncMock()
    c.close = AsyncMock()
    return c


@pytest.fixture
def make_guardrail(mock_client):
    """Build a guardrail wired to `mock_client`, with extra ctor kwargs."""

    def _make(**kwargs):
        _install_sdk_stub(client_factory=lambda **kw: mock_client)
        from wonderfence_guardrail import WonderFenceGuardrail

        g = WonderFenceGuardrail(
            guardrail_name="wf-test",
            api_key="test-key",
            app_id="test-app",
            **kwargs,
        )
        g._client_cache["test-key"] = mock_client
        return g

    return _make


@pytest.fixture
def guardrail(make_guardrail):
    return make_guardrail()


@pytest.fixture(autouse=True)
def _restore_guardrail_logger():
    """Undo _configure_logging's mutations — the logger is module-level state."""
    from wonderfence_guardrail import logger

    level, handlers, propagate = logger.level, logger.handlers[:], logger.propagate
    yield
    logger.setLevel(level)
    logger.handlers[:] = handlers
    logger.propagate = propagate
