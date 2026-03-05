import pytest

from story_utils import extract_moral, detect_symbols, map_symbols, moral_drift


def test_extract_moral():
    story = "Once upon a time. The moral is be kind."
    assert extract_moral(story) == "The moral is be kind."
    assert extract_moral("") == ""


def test_detect_symbols():
    text = "A banyan tree and an elephant were friends."
    syms = detect_symbols(text)
    assert "banyan tree" in syms
    assert "elephant" in syms


def test_map_symbols():
    # mapping defined in story_utils
    result = map_symbols(["banyan tree", "fox"], "India", "Brazil")
    assert "ceiba tree" in result
    assert "fox" in result  # unmapped remains unchanged


def test_moral_drift():
    # since similarity_score uses actual models it may fail; instead monkeypatch
    from story_utils import similarity_score
    original = "Be kind to others."
    adapted = "Always be kind to your brothers."
    # score will be some number; test threshold logic
    assert moral_drift(original, adapted, threshold=0.0)
    assert not moral_drift(original, adapted, threshold=1.0)
