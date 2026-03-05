# # app.py - Lingoo: Cultural Story Adaptation Engine
import os
import streamlit as st
from brain import adapt_story, get_backend_info
from culture_detector import detect_culture

# Page Config
st.set_page_config(
    page_title="Lingoo – Cultural Story Adaptation",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# (loaded from style.css)
def _load_css(path: str) -> None:
    with open(path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css(os.path.join(os.path.dirname(__file__), "style.css"))

# Culture / Region Data

CULTURES = {
    "── Countries ──": None,   # group header (disabled)
    "India":      "India",
    "Brazil":     "Brazil",
    "Nigeria":    "Nigeria",
    "Mexico":     "Mexico",
    "Japan":      "Japan",
    "China":      "China",
    "Egypt":      "Egypt",
    "Germany":    "Germany",
    "Indonesia":  "Indonesia",
    "Kenya":      "Kenya",
    "── Indian States ──": None,
    "Tamil Nadu":    "Tamil Nadu",
    "Kerala":        "Kerala",
    "Karnataka":     "Karnataka",
    "Maharashtra":   "Maharashtra",
    "Gujarat":       "Gujarat",
    "Rajasthan":     "Rajasthan",
    "Punjab":        "Punjab",
    "West Bengal":   "West Bengal",
    "Andhra Pradesh":"Andhra Pradesh",
    "Telangana":     "Telangana",
    "Odisha":        "Odisha",
    "Assam":         "Assam",
    "Goa":           "Goa",
}

# Selectable options only (no headers)
CULTURE_OPTIONS = [k for k, v in CULTURES.items() if v is not None]

# Sidebar 
with st.sidebar:
    st.markdown('<div class="sidebar-title">📖 Lingoo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Cross-Cultural Story Adaptation Engine</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Backend**")
    backend_info = get_backend_info()
    st.markdown(f'<div class="backend-chip">{backend_info}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        '<div class="sidebar-sub">'
        "1. Extract moral &amp; symbols<br>"
        "2. Fetch Wikidata cultural context<br>"
        "3. LLM rewrites the story<br>"
        "4. Semantic similarity scored"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        st.markdown(
            '<div class="sidebar-sub" style="color:rgba(244,165,122,0.8) !important;">'
            "⚠️ No HF_TOKEN set. API calls may be rate-limited on the free tier.<br><br>"
            "Set: <code>export HF_TOKEN=hf_...</code>"
            "</div>",
            unsafe_allow_html=True,
        )

# Hero 
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">✦ AI-Powered Cultural Adaptation</div>
    <div class="hero-title">Lingoo</div>
    <div class="hero-sub">Preserve meaning &nbsp;·&nbsp; Bridge worlds &nbsp;·&nbsp; Every story belongs everywhere</div>
</div>
""", unsafe_allow_html=True)

# Steps 
st.markdown("""
<div class="steps">
    <div class="step"><div class="step-num">1</div><div class="step-lbl">Input Story &amp; Target Culture</div></div>
    <div class="step"><div class="step-num">2</div><div class="step-lbl">Extract Moral &amp; Symbols</div></div>
    <div class="step"><div class="step-num">3</div><div class="step-lbl">Map Cultural Elements</div></div>
    <div class="step"><div class="step-num">4</div><div class="step-lbl">AI Rewrites &amp; Scores</div></div>
</div>
""", unsafe_allow_html=True)

# Layout 
left, right = st.columns([1.1, 1], gap="large")

with left:
    # Settings card
    st.markdown('<div class="card"><div class="card-title">🌍 Adaptation Settings</div>', unsafe_allow_html=True)

    culture = st.selectbox(
        "Target Culture / Region",
        CULTURE_OPTIONS,
        help="Choose a country or an Indian state as the cultural target.",
    )

    age = st.number_input(
        "Target Age",
        min_value=5, max_value=18, value=8,
        help="The language complexity will be adjusted for this age group.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Story input card
    st.markdown('<div class="card"><div class="card-title">📜 Original Folktale</div>', unsafe_allow_html=True)

    story_input = st.text_area(
        label="story",
        label_visibility="collapsed",
        height=260,
        placeholder=(
            "Once upon a time, in a village beneath a great banyan tree, "
            "a wise tortoise taught the children the value of patience…"
        ),
    )

    # Auto-detect culture hint
    if story_input.strip():
        detected_name, detected_type = detect_culture(story_input)
        if detected_name:
            st.markdown(
                f'<div class="chip">🔍 Detected in story: <strong>{detected_name}</strong>'
                f'&nbsp;({detected_type})</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    transform_btn = st.button("✦  Transform Story", use_container_width=True)

with right:
    if transform_btn:
        if not story_input.strip():
            st.warning("Please enter a story first.")
        else:
            with st.spinner(f"Adapting for {culture} · Age {age}…"):
                try:
                    adapted, m_score, s_score, moral, symbols = adapt_story(story_input, culture, age)
                except Exception as e:
                    st.error(f"Error during adaptation: {e}")
                    st.stop()

            # Moral & symbols
            if moral or symbols:
                if moral:
                    st.markdown(
                        f'<div class="info-card">'
                        f'<div class="info-label">💡 Extracted Moral</div>'
                        f'<div class="info-value">{moral}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                if symbols:
                    st.markdown(
                        f'<div class="info-card">'
                        f'<div class="info-label">🔑 Cultural Symbols Identified</div>'
                        f'<div class="info-value">{symbols}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Adapted story
            st.markdown(
                f'<div class="result-box">'
                f'<div class="result-title">📖 Adapted for {culture} &nbsp;·&nbsp; Age {age}</div>'
                f'<div class="result-text">{adapted}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Score bars ──
            def _score_bar(pct: int, label: str, sublabel: str, color_start: str, color_end: str) -> str:
                return (
                    f'<div class="score-wrap">'
                    f'  <div style="min-width:56px">'
                    f'    <div class="score-num">{pct}%</div>'
                    f'    <div class="score-lbl">{label}</div>'
                    f'  </div>'
                    f'  <div style="flex:1">'
                    f'    <div class="bar-outer">'
                    f'      <div class="bar-inner" style="width:{pct}%;background:linear-gradient(90deg,{color_start},{color_end})"></div>'
                    f'    </div>'
                    f'    <div class="score-sublbl">{sublabel}</div>'
                    f'  </div>'
                    f'</div>'
                )

            if m_score is not None:
                mp  = int(float(m_score) * 100)
                mvd = "Excellent" if mp >= 75 else "Good" if mp >= 55 else "Needs work"
                st.markdown(
                    _score_bar(mp, f"Moral Preservation · {mvd}",
                               "Did the lesson survive the cultural rewrite? Target: 75%+",
                               "#F4A57A", "#5C1A2E"),
                    unsafe_allow_html=True,
                )

            if s_score is not None:
                sp  = int(float(s_score) * 100)
                svd = "High overlap" if sp >= 65 else "Healthy distance" if sp >= 35 else "Very different"
                st.markdown(
                    _score_bar(sp, f"Semantic Similarity · {svd}",
                               "How similar are the two stories overall? 40–65% is the sweet spot for cultural adaptation.",
                               "#A0C4FF", "#3A6EA5"),
                    unsafe_allow_html=True,
                )
    else:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:3rem;margin-bottom:1rem;">📖</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.25rem;
                        color:#5C1A2E;font-weight:700;margin-bottom:0.5rem;">
                Your adaptation will appear here
            </div>
            <div style="font-size:0.88rem;color:#9A6070;line-height:1.7;max-width:270px;">
                Paste a folktale, choose a culture and age group,<br>
                then click <strong>Transform Story</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer 
st.markdown("""
<hr>
<div style="text-align:center;font-size:0.78rem;color:#9A6070;padding:0.4rem 0 1rem;">
    Lingoo &nbsp;·&nbsp; Cross-Cultural Story Adaptation &nbsp;·&nbsp;
    Qwen2.5 · Wikidata · SentenceTransformers
</div>
""", unsafe_allow_html=True)