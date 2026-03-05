import pytest

import brain


def test_adapt_story_structure(monkeypatch):
    # stub generate_text so we don't need models
    monkeypatch.setattr(brain, "generate_text", lambda prompt: "adapted version")
    # stub similarity_score to avoid loading models
    monkeypatch.setattr(brain, "similarity_score", lambda a, b: 0.8)

    original = "A simple story"
    result = brain.adapt_story(original, "Brazil", 7, original_culture="India")
    assert isinstance(result, dict)
    expected_keys = {"adapted", "score", "moral", "symbols", "mapped_symbols", "drift_ok"}
    assert expected_keys.issubset(result.keys())
    assert result["adapted"] == "adapted version"
    assert result["score"] == 0.8
