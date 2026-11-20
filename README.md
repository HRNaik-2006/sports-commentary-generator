# Sports Commentary Generator

A local Streamlit drill project for adapting a tiny language model with a handwritten PyTorch LoRA module and generating context-aware play-by-play snippets. It runs without a pretrained-model download or external API.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

Use **Train / load adapter** in the sidebar, then submit a match event. The interface caches the trained adapter for the current Streamlit session. Generated snippets are saved as JSONL in `data/generations.jsonl`.

## How the LoRA drill works

`app/llm.py` defines a small character-level next-token model. Its output projection is a frozen base linear layer plus `B(A(x)) × alpha/r`; the adapter matrices are created directly in PyTorch and only those matrices receive optimizer updates. Seed examples span soccer, basketball, hockey, football, and tennis. The intentionally compact model is for learning and prototyping, not broadcast-ready language quality.

## Customize

Add prompt/commentary pairs in `app/core.py`'s `seed_examples()`. Tune the rank, learning rate, number of steps, and sampling temperature in `app/llm.py` and `app/main.py`. The JSONL helpers in `app/storage.py` can also persist app-specific records.

## Docker

```bash
docker build -t sports-commentary-generator .
docker run --rm -p 8501:8501 sports-commentary-generator
```

## Tests

```bash
pytest -v
```
