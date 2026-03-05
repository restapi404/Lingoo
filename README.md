# Lingoo – Cultural Story Adaptation Engine

This repository contains a prototype system that adapts folktales and
educational stories for different cultures and age groups by extracting morals
and cultural symbols, mapping them to equivalent concepts, and rewriting the
narrative with a large language model.

## Directory layout

```
Lingoo/
├── app.py            # Streamlit frontend
├── brain.py          # Core logic (model loading, adaptation)
├── culture_detector.py
├── story_utils.py    # moral extraction, symbol mapping, scoring
├── wikidata_fetch.py # Wikidata helpers
├── tests/            # unit tests
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment (Python 3.10+ recommended):
   ```bash
   python -m venv lingua_env
   lingua_env\Scripts\activate  # Windows
   # or: source lingua_env/bin/activate  # macOS/Linux
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note:* you may need to install a compatible `torch` wheel manually if the
   automatic installation fails. For GPU support choose the appropriate CUDA
   build.

## Running the demo

Start the Streamlit app:

```bash
streamlit run app.py
```

Select a target culture and age, paste a story (or load an example), and click
`Transform Story`. The UI will display the adapted narrative along with
analysis information.

## Testing

Run the unit tests with `pytest` (install it if necessary):

```bash
pip install pytest
pytest
```

The tests exercise the culture detector, utility functions, and the
`adapt_story` interface (with model calls stubbed).

## Extending the project

- Improve the `story_utils` functions with real NLP or knowledge graphs.
- Expand the cultural symbol database and mapping engine.
- Add moral drift detection using a fine-tuned classifier.
- Integrate with external EdTech platforms or mobile apps.

## Notes for presentation

* The frontend demonstrates full flow from user input to adapted output.
* Symbol mapping and drift detection are currently simplistic stubs; the
  architecture supports swapping in more advanced components.
* The code is structured to allow easy unit testing and future refactoring.

---

Good luck with your submission!
