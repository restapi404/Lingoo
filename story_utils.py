# #story_utils.py – Utility functions for story analysis and adaptation.

import re

# we import the similarity scoring lazily to avoid circular dependencies


def similarity_score(text1, text2):
    """Return cosine similarity between two pieces of text.

    This helper grabs the embedding model from :mod:`brain` on demand so that
    the module can be imported without pulling in heavy dependencies at
    import-time.
    """
    try:
        from brain import _ensure_models
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        # if the import fails, behave gracefully in tests or when models aren't
        # available; return zero so nothing accidentally passes a threshold.
        return 0.0

    _, _, embed_model = _ensure_models()
    emb1 = embed_model.encode([text1])
    emb2 = embed_model.encode([text2])
    return cosine_similarity(emb1, emb2)[0][0]


def extract_moral(text):
    """Naively extract a moral from a story.

    The current implementation simply returns the last sentence.
    Future versions could employ an NLP pipeline or a small LLM prompt.
    """
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return sentences[-1] if sentences else ""


def detect_symbols(text):
    """Find a few culture-specific keywords in the text.

    This is a placeholder for a more sophisticated symbol detector.  It keeps
    a small dictionary of tokens and returns those that appear.  The list can
    be expanded as needed.
    """
    tokens = [
        "banyan tree",
        "elephant",
        "river",
        "fox",
        "monkey",
        "jaguar",
        "ceiba",
        "temple",
    ]
    found = []
    lower = text.lower()
    for tok in tokens:
        if tok in lower:
            found.append(tok)
    return found


# simple hand-crafted symbol map for demonstration purposes
_SYMBOL_MAP = {
    ("banyan tree", "India", "Brazil"): "ceiba tree",
    ("elephant", "India", "Brazil"): "jaguar",
}


def map_symbols(symbols, src_culture, tgt_culture):
    """Map a list of source symbols to equivalents in the target culture."""
    out = []
    for sym in symbols:
        mapped = _SYMBOL_MAP.get((sym, src_culture, tgt_culture), sym)
        out.append(mapped)
    return out


def moral_drift(original_moral, adapted_text, threshold=0.7):
    """Determine whether the moral has drifted after adaptation.

    A very simple implementation that compares the original moral to the
    adapted text using a semantic score.  Returns True when drift is *not*
    detected (score >= threshold).
    """
    if not original_moral:
        return False
    score = similarity_score(original_moral, adapted_text)
    return score >= threshold
