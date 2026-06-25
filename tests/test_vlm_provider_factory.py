import pytest

from src.vlm.local_gemma_provider import LocalGemmaProvider
from src.vlm.mock_provider import MockVLMProvider
from src.vlm.provider_factory import build_vlm_provider, resolve_vlm_provider
from src.vlm.qwen_provider import QwenProvider


def test_provider_factory_selects_mock_qwen_and_local_gemma():
    assert isinstance(build_vlm_provider(vlm_provider="mock", vlm_mode="local"), MockVLMProvider)
    assert isinstance(build_vlm_provider(vlm_provider="qwen", vlm_mode="web"), QwenProvider)
    assert isinstance(build_vlm_provider(vlm_provider="auto", vlm_mode="local"), LocalGemmaProvider)
    assert isinstance(build_vlm_provider(vlm_provider="gemma", vlm_mode="local"), LocalGemmaProvider)


def test_cli_auto_provider_resolution():
    assert resolve_vlm_provider("auto", "web") == "qwen"
    assert resolve_vlm_provider("auto", "local") == "gemma"
    assert resolve_vlm_provider("mock", "local") == "mock"


def test_vlm_mode_provider_conflicts_are_rejected():
    with pytest.raises(ValueError, match="qwen requires"):
        resolve_vlm_provider("qwen", "local")
    with pytest.raises(ValueError, match="gemma requires"):
        resolve_vlm_provider("gemma", "web")
    with pytest.raises(ValueError, match="qwen requires"):
        build_vlm_provider(vlm_provider="qwen", vlm_mode="local")
    with pytest.raises(ValueError, match="gemma requires"):
        build_vlm_provider(vlm_provider="gemma", vlm_mode="web")
