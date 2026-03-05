# # culture_detector.py – Detect cultural origin of a story from its text.


import re

# Country keyword map 
COUNTRIES: dict[str, str] = {
    "india":       "India",
    "indian":      "India",
    "brazil":      "Brazil",
    "brazilian":   "Brazil",
    "nigeria":     "Nigeria",
    "nigerian":    "Nigeria",
    "mexico":      "Mexico",
    "mexican":     "Mexico",
    "japan":       "Japan",
    "japanese":    "Japan",
    "china":       "China",
    "chinese":     "China",
    "egypt":       "Egypt",
    "egyptian":    "Egypt",
    "germany":     "Germany",
    "german":      "Germany",
    "indonesia":   "Indonesia",
    "indonesian":  "Indonesia",
    "kenya":       "Kenya",
    "kenyan":      "Kenya",
    "france":      "France",
    "french":      "France",
    "usa":         "USA",
    "america":     "USA",
    "american":    "USA",
    "greece":      "Greece",
    "greek":       "Greece",
    "korea":       "Korea",
    "korean":      "Korea",
    "russia":      "Russia",
    "russian":     "Russia",
    "peru":        "Peru",
    "peruvian":    "Peru",
    "ethiopia":    "Ethiopia",
    "ethiopian":   "Ethiopia",
    "vietnam":     "Vietnam",
    "vietnamese":  "Vietnam",
}

# Indian state keyword map 
INDIAN_STATES: dict[str, str] = {
    "tamil nadu":     "Tamil Nadu",
    "tamil":          "Tamil Nadu",
    "kerala":         "Kerala",
    "karnataka":      "Karnataka",
    "kannada":        "Karnataka",
    "maharashtra":    "Maharashtra",
    "marathi":        "Maharashtra",
    "gujarat":        "Gujarat",
    "gujarati":       "Gujarat",
    "rajasthan":      "Rajasthan",
    "rajasthani":     "Rajasthan",
    "punjab":         "Punjab",
    "punjabi":        "Punjab",
    "west bengal":    "West Bengal",
    "bengali":        "West Bengal",
    "andhra":         "Andhra Pradesh",
    "telugu":         "Telangana / Andhra",
    "telangana":      "Telangana",
    "odisha":         "Odisha",
    "odia":           "Odisha",
    "assam":          "Assam",
    "assamese":       "Assam",
    "bihar":          "Bihar",
    "bihari":         "Bihar",
    "goa":            "Goa",
    "konkani":        "Goa",
    "uttarakhand":    "Uttarakhand",
    "himachal":       "Himachal Pradesh",
    "kashmir":        "Kashmir",
}

#  Cultural symbol hints 
# Maps a recognisable symbol → (culture, display_type)
SYMBOL_HINTS: dict[str, tuple[str, str]] = {
    # India / Indian states
    "banyan tree":   ("India",        "symbol"),
    "banyan":        ("India",        "symbol"),
    "peepal":        ("India",        "symbol"),
    "diwali":        ("India",        "symbol"),
    "holi":          ("India",        "symbol"),
    "ganesh":        ("India",        "symbol"),
    "lotus":         ("India",        "symbol"),
    "cobra":         ("India",        "symbol"),
    "mongoose":      ("India",        "symbol"),
    "peacock":       ("India",        "symbol"),
    "sitar":         ("India",        "symbol"),
    "kolam":         ("Tamil Nadu",   "symbol"),
    "pongal":        ("Tamil Nadu",   "symbol"),
    "onam":          ("Kerala",       "symbol"),
    "kathakali":     ("Kerala",       "symbol"),
    "yakshagana":    ("Karnataka",    "symbol"),
    "garba":         ("Gujarat",      "symbol"),
    "durga puja":    ("West Bengal",  "symbol"),
    "bihu":          ("Assam",        "symbol"),
    # Brazil
    "amazon":        ("Brazil",       "symbol"),
    "samba":         ("Brazil",       "symbol"),
    "carnival":      ("Brazil",       "symbol"),
    "capoeira":      ("Brazil",       "symbol"),
    "piranha":       ("Brazil",       "symbol"),
    "jaguar":        ("Brazil",       "symbol"),
    "toucan":        ("Brazil",       "symbol"),
    # Nigeria
    "baobab":        ("Nigeria/Kenya","symbol"),
    "anansi":        ("Nigeria",      "symbol"),
    "yoruba":        ("Nigeria",      "symbol"),
    "igbo":          ("Nigeria",      "symbol"),
    "hausa":         ("Nigeria",      "symbol"),
    # Mexico
    "sombrero":      ("Mexico",       "symbol"),
    "mariachi":      ("Mexico",       "symbol"),
    "quetzal":       ("Mexico",       "symbol"),
    "aztec":         ("Mexico",       "symbol"),
    "maya":          ("Mexico",       "symbol"),
    "día de muertos":("Mexico",       "symbol"),
    # Japan
    "samurai":       ("Japan",        "symbol"),
    "ninja":         ("Japan",        "symbol"),
    "geisha":        ("Japan",        "symbol"),
    "cherry blossom":("Japan",        "symbol"),
    "sakura":        ("Japan",        "symbol"),
    "torii":         ("Japan",        "symbol"),
    "tanuki":        ("Japan",        "symbol"),
    "kitsune":       ("Japan",        "symbol"),
    # China
    "dragon":        ("China",        "symbol"),
    "pagoda":        ("China",        "symbol"),
    "panda":         ("China",        "symbol"),
    "jade":          ("China",        "symbol"),
    "lantern festival":("China",      "symbol"),
    # Egypt
    "pharaoh":       ("Egypt",        "symbol"),
    "pyramid":       ("Egypt",        "symbol"),
    "sphinx":        ("Egypt",        "symbol"),
    "nile":          ("Egypt",        "symbol"),
    "hieroglyph":    ("Egypt",        "symbol"),
    # Kenya
    "savanna":       ("Kenya",        "symbol"),
    "safari":        ("Kenya",        "symbol"),
    "maasai":        ("Kenya",        "symbol"),
    "lion":          ("Kenya",        "symbol"),  # loose hint
    "elephant":      ("Kenya",        "symbol"),
    # Germany
    "oktoberfest":   ("Germany",      "symbol"),
    "autobahn":      ("Germany",      "symbol"),
    "pretzel":       ("Germany",      "symbol"),
    # Greece
    "acropolis":     ("Greece",       "symbol"),
    "zeus":          ("Greece",       "symbol"),
    "olympus":       ("Greece",       "symbol"),
    "odyssey":       ("Greece",       "symbol"),
    # Korea
    "kimchi":        ("Korea",        "symbol"),
    "hanbok":        ("Korea",        "symbol"),
    "hanok":         ("Korea",        "symbol"),
}


def detect_culture(text: str) -> tuple[str | None, str | None]:
    """
    Scan story text for cultural markers.
    Returns (culture_name, detection_type) or (None, None).

    Priority: Indian state keywords > country keywords > symbol hints.
    """
    lowered = text.lower()

    # 1. Indian states (most specific)
    for kw, state in INDIAN_STATES.items():
        if re.search(r'\b' + re.escape(kw) + r'\b', lowered):
            return state, "Indian state"

    # 2. Country keywords
    for kw, country in COUNTRIES.items():
        if re.search(r'\b' + re.escape(kw) + r'\b', lowered):
            return country, "country"

    # 3. Cultural symbol hints (substring match — symbols can be multi-word)
    for symbol, (culture, _) in SYMBOL_HINTS.items():
        if symbol in lowered:
            return culture, f'symbol: "{symbol}"'

    return None, None
