# ERA Translation Agent

Enterprise-focused multilingual translation pipeline. Translates text or documents between 30 languages, then automatically verifies its own translation quality via back-translation and retries with a different decoding strategy if confidence is low.

## How it works

1. Translate the input with NLLB-200.
2. Translate the result *back* to the source language.
3. Compare the original and the round-trip translation using LaBSE (a multilingual sentence-embedding model) — semantic similarity, not just word overlap.
4. If similarity is below `0.85`, retry with a different decoding strategy (beam search → sampling → wider beam search), up to 3 attempts, keeping the best-scoring result.
5. Cache high-quality translations (SQLite) so repeated segments are instant.

## Why NLLB + LaBSE, not COMET-Kiwi?

An earlier version of this project used COMET-Kiwi for quality scoring. It was dropped for two reasons:

- **It's a gated Hugging Face model** — requires a Hugging Face account, logging in, and manually accepting a license before it will download. Not something a recruiter running this repo should have to do.
- **Fragile dependency chain** — `comet` pulls in `pytorch_lightning` → `torchmetrics`, which depends on `pkg_resources`, a module `setuptools` has been removing. This broke on a clean install with no warning until runtime.

LaBSE is open-access, has no license gate, and does the same job (measuring whether two texts mean the same thing) with far fewer moving parts.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Set `GRADIO_SHARE=true` to get a public share link (off by default). No API keys or Hugging Face login required — both models are open-access and download automatically on first use.

## Features

- **Translation** — text or PDF/DOCX/TXT upload, with back-translation shown alongside quality/consistency/combined scores
- **Translation Memory** — searchable cache of past translations
- **Glossary** — pin domain-specific terms to a fixed translation
- **Analytics** — usage stats and quality trends over time

## Evaluation

`analysis/evaluation.ipynb` benchmarks the pipeline on a small multilingual test set and analyzes the pipeline's own logged usage history. Run it after using the app for a while to get real, measured numbers.

## Tech stack

Python, PyTorch, Hugging Face Transformers, Sentence-Transformers (LaBSE), Gradio, SQLite.

## License

MIT
