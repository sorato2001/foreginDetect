import pytest

from src.pipeline.analyze_event import _resolve_vlm_provider
from src.vlm.local_gemma_provider import LocalGemmaProvider
from src.vlm.mock_provider import MockVLMProvider
from src.vlm.provider_factory import build_vlm_provider
from src.vlm.qwen_provider import QwenProvider


def test_provider_factory_selects_mock_qwen_and_local_gemma():
    assert isinstance(build_vlm_provider(vlm_provider="mock", vlm_mode="local"), MockVLMProvider)
    assert isinstance(build_vlm_provider(vlm_provider="qwen", vlm_mode="web"), QwenProvider)
    assert isinstance(build_vlm_provider(vlm_provider="auto", vlm_mode="local"), LocalGemmaProvider)
    assert isinstance(build_vlm_provider(vlm_provider="gemma", vlm_mode="local"), LocalGemmaProvider)


def test_cli_auto_provider_resolution():
    assert _resolve_vlm_provider("auto", "web") == "qwen"
    assert _resolve_vlm_provider("auto", "local") == "gemma"
    assert _resolve_vlm_provider("mock", "local") == "mock"


def test_vlm_mode_provider_conflicts_are_rejected():
    with pytest.raises(ValueError, match="qwen requires"):
        _resolve_vlm_provider("qwen", "local")
    with pytest.raises(ValueError, match="gemma requires"):
        _resolve_vlm_provider("gemma", "web")
    with pytest.raises(ValueError, match="Qwen provider requires"):
        build_vlm_provider(vlm_provider="qwen", vlm_mode="local")
    with pytest.raises(ValueError, match="Gemma/local_gemma provider requires"):
        build_vlm_provider(vlm_provider="gemma", vlm_mode="web")
