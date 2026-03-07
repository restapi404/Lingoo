"""
brain.py – Core adaptation engine for Lingoo.

Device strategy (auto-detected at startup):
  ┌─ CUDA / MPS GPU available?
  │     YES → Load Qwen2.5-3B-Instruct locally (4-bit quantised on CUDA)
  │     NO  → HuggingFace Inference API
  │               Primary : Qwen/Qwen2.5-3B-Instruct  (serverless)
  │               Fallback: mistralai/Mistral-7B-Instruct-v0.3
  └─ Embedding model always runs on CPU (all-MiniLM-L6-v2)

Environment variables (optional):
  HF_TOKEN          – HuggingFace token (avoids rate-limits on API path)
"""

import os
import re
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from wikidata_fetch import search_wikidata, fetch_wikidata, extract_culture_info

# ─────────────────────────────────────────────────────────────────────────────
# 1. Device Detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

DEVICE     = _detect_device()
USE_LOCAL  = DEVICE in ("cuda", "mps")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Embedding Model  (always CPU)
# ─────────────────────────────────────────────────────────────────────────────

_embed_model = None

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
    return _embed_model


# ─────────────────────────────────────────────────────────────────────────────
# 3a. LOCAL path  – Qwen2.5-3B-Instruct with BitsAndBytes 4-bit (GPU only)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_local_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
    )
    return tokenizer, model


def _generate_local(prompt: str, max_new_tokens: int = 700) -> str:
    import torch
    tokenizer, model = _load_local_model()

    messages = [
        {"role": "system", "content": "You are Lingoo, an expert in cross-cultural storytelling."},
        {"role": "user",   "content": prompt},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    # Strip the prompt tokens from the output
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 3b. API path  – HuggingFace Inference API (CPU / no GPU)
# ─────────────────────────────────────────────────────────────────────────────

HF_PRIMARY     = "google/gemma-3-4b-it"
HF_FALLBACK    = "google/gemma-3-1b-it"


def _hf_headers() -> dict:
    """Legacy function - no longer used with InferenceClient."""
    return {}


def _generate_hf(prompt: str, model: str, max_new_tokens: int = 700) -> str:
    """Generate text using HuggingFace InferenceClient."""
    from huggingface_hub import InferenceClient
    
    client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))
    
    try:
        message = client.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are Lingoo, an expert in cross-cultural storytelling."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        return message.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"HF client failed for model '{model}': {e}")


def _generate_api(prompt: str, max_new_tokens: int = 700) -> str:
    """Try Qwen on HF first; fall back to Mistral if it fails."""
    try:
        return _generate_hf(prompt, HF_PRIMARY, max_new_tokens)
    except Exception as e1:
        try:
            return _generate_hf(prompt, HF_FALLBACK, max_new_tokens)
        except Exception as e2:
            raise RuntimeError(
                f"Both HF models failed.\n"
                f"  Qwen  : {e1}\n"
                f"  Mistral: {e2}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Unified dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def generate_text(prompt: str, max_new_tokens: int = 700) -> str:
    if USE_LOCAL:
        return _generate_local(prompt, max_new_tokens)
    return _generate_api(prompt, max_new_tokens)


def get_backend_info() -> str:
    """Return a short string shown in the UI sidebar."""
    if USE_LOCAL:
        return f"🖥️ Local  ·  Qwen2.5-3B-Instruct  ({DEVICE.upper()})"
    has_token = bool(os.environ.get("HF_TOKEN"))
    auth_note = "authenticated" if has_token else "anonymous – rate-limited"
    return f"☁️ HF API  ·  Qwen2.5-3B  ({auth_note})"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Moral Preservation Score
# ─────────────────────────────────────────────────────────────────────────────

def _embed(texts: list[str]):
    return _get_embed_model().encode(texts)


def _extract_moral_short(story: str) -> str:
    """
    Quick single-sentence moral extraction used only for scoring.
    Avoids calling the heavy LLM again — uses a lightweight heuristic prompt.
    """
    prompt = (
        "In one sentence only, state the core moral lesson of this story.\n"
        "Output only the sentence, nothing else.\n\n"
        f"STORY:\n{story}"
    )
    try:
        return generate_text(prompt, max_new_tokens=60).strip()
    except Exception:
        # Fallback: use the first 200 chars as a proxy
        return story[:200]


def semantic_similarity_score(original: str, adapted: str) -> float:
    """
    Pure cosine similarity between the full story embeddings.

    Measures how much the adapted story *resembles* the original in
    overall meaning and wording. This will naturally be lower after a
    cultural rewrite — that's expected and even desirable.
    A score of 40–65% is healthy: the story is clearly different in
    cultural surface but still thematically related.
    """
    m    = _get_embed_model()
    emb1 = m.encode([original])
    emb2 = m.encode([adapted])
    return round(float(cosine_similarity(emb1, emb2)[0][0]), 4)


def moral_preservation_score(original: str, adapted: str, original_moral: str = "") -> float:
    """
    How well the moral/lesson survived the cultural rewrite.

    Strategy (weighted blend):
      60% – cosine similarity of the extracted moral sentences from each story
      40% – full-story thematic similarity (safety net)

    Comparing morals directly avoids the false-low scores caused by
    surface-level wording changes during cultural adaptation.
    A score of 75%+ means the lesson was preserved well.
    """
    m = _get_embed_model()

    orig_moral = original_moral if original_moral else _extract_moral_short(original)
    adpt_moral = _extract_moral_short(adapted)

    emb_orig_m = m.encode([orig_moral])
    emb_adpt_m = m.encode([adpt_moral])
    moral_sim  = float(cosine_similarity(emb_orig_m, emb_adpt_m)[0][0])

    emb_orig_s = m.encode([original])
    emb_adpt_s = m.encode([adapted])
    story_sim  = float(cosine_similarity(emb_orig_s, emb_adpt_s)[0][0])

    return round(0.6 * moral_sim + 0.4 * story_sim, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cultural Context from Wikidata
# ─────────────────────────────────────────────────────────────────────────────

def build_culture_prompt(culture: str) -> str:
    wid  = search_wikidata(culture)
    base = f"Culture: {culture}\nDescription: A rich cultural tradition with its own symbols, festivals, and stories."
    if not wid:
        return base

    data = fetch_wikidata(wid)
    if not data:
        return base

    info = extract_culture_info(data)
    name = info.get("name", culture)
    desc = info.get("description", "")
    return f"Culture: {name}\nDescription: {desc}" if desc else base


# ─────────────────────────────────────────────────────────────────────────────
# 7. Step 1 – Extract Moral & Symbols
# ─────────────────────────────────────────────────────────────────────────────

def extract_moral_and_symbols(story: str) -> tuple[str, str]:
    prompt = (
        "Analyze the following folktale carefully.\n"
        "Reply in EXACTLY this two-line format:\n"
        "MORAL: <core life lesson in 1-2 sentences>\n"
        "SYMBOLS: <comma-separated culture-specific symbols, animals, places, foods, objects>\n\n"
        f"STORY:\n{story}"
    )
    try:
        raw           = generate_text(prompt, max_new_tokens=200)
        moral_m       = re.search(r'MORAL:\s*(.+?)(?:\n|SYMBOLS:|$)', raw, re.DOTALL)
        symbols_m     = re.search(r'SYMBOLS:\s*(.+?)$',               raw, re.DOTALL)
        moral         = moral_m.group(1).strip()   if moral_m   else ""
        symbols       = symbols_m.group(1).strip() if symbols_m else ""
        return moral, symbols
    except Exception:
        return "", ""


# ─────────────────────────────────────────────────────────────────────────────
# 8. Step 2 – Rewrite Story
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_story(
    original_story: str,
    moral: str,
    symbols: str,
    culture_context: str,
    target_age: int,
) -> str:
    if target_age <= 7:
        age_note = "Very simple words, short sentences, warm and bedtime-story-like."
    elif target_age <= 11:
        age_note = "Clear, engaging language with some descriptive detail (early reader level)."
    else:
        age_note = "Descriptive, emotionally rich language appropriate for young adults."

    prompt = (
        "You are Lingoo, a cross-cultural storytelling expert.\n"
        "Rewrite the folktale below so it feels completely native to the target culture "
        "while preserving the moral exactly.\n\n"
        f"=== ORIGINAL STORY ===\n{original_story}\n\n"
        f"=== CORE MORAL ===\n{moral or 'Preserve the original moral.'}\n\n"
        f"=== ORIGINAL CULTURAL SYMBOLS ===\n{symbols or 'Identify and replace as needed.'}\n\n"
        f"=== TARGET CULTURE ===\n{culture_context}\n\n"
        "=== RULES ===\n"
        "1. Replace symbols (places, animals, foods, festivals, names) with authentic "
        "   equivalents from the target culture.\n"
        "2. Preserve the moral and emotional arc EXACTLY — do NOT change the lesson.\n"
        "3. The story must feel like it originated from the target culture.\n"
        f"4. Writing style: {age_note}\n"
        "5. Length: 150–350 words.\n"
        "6. Output ONLY the story. No title, no preamble, no explanation.\n"
    )
    return generate_text(prompt, max_new_tokens=700)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def adapt_story(
    original_text: str,
    target_culture: str,
    target_age: int,
) -> tuple[str, float | None, float | None, str, str]:
    """
    Returns:
        (adapted_story, moral_score, semantic_score, moral, symbols)

        moral_score    – how well the lesson/message was preserved (target: 75%+)
        semantic_score – how similar the two stories are overall (expect 40–65%)
    """
    moral, symbols  = extract_moral_and_symbols(original_text)
    culture_context = build_culture_prompt(target_culture)
    adapted         = rewrite_story(original_text, moral, symbols, culture_context, target_age)

    try:
        m_score = moral_preservation_score(original_text, adapted, original_moral=moral)
    except Exception:
        m_score = None

    try:
        s_score = semantic_similarity_score(original_text, adapted)
    except Exception:
        s_score = None

    return adapted, m_score, s_score, moral, symbols
