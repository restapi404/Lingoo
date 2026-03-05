# # wikidata_fetch.py – Fetch and extract cultural context from Wikidata.

import requests

SEARCH_URL = "https://www.wikidata.org/w/api.php"
ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Lingoo-StoryAdapter/1.0 (educational project)"})


def search_wikidata(entity_name: str) -> str | None:
    """Search Wikidata for an entity. Returns QID string or None."""
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": entity_name,
        "limit": 3,
    }
    try:
        resp = _SESSION.get(SEARCH_URL, params=params, timeout=8)
        resp.raise_for_status()
        results = resp.json().get("search", [])
        if results:
            return results[0]["id"]
    except Exception:
        pass
    return None


def fetch_wikidata(entity_id: str) -> dict | None:
    """Fetch full entity JSON from Wikidata by QID."""
    try:
        resp = _SESSION.get(ENTITY_URL.format(entity_id), timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def extract_culture_info(data: dict) -> dict:
    """
    Extract name and description from a Wikidata entity response.
    Returns: { "name": str, "description": str, "aliases": list[str] }
    """
    empty = {"name": "", "description": "", "aliases": []}
    if not data or "entities" not in data:
        return empty
    try:
        entity = list(data["entities"].values())[0]
        name   = entity.get("labels",       {}).get("en", {}).get("value", "")
        desc   = entity.get("descriptions", {}).get("en", {}).get("value", "")
        aliases = [
            a["value"]
            for a in entity.get("aliases", {}).get("en", [])
        ]
        return {"name": name, "description": desc, "aliases": aliases}
    except Exception:
        return empty
