Context-Aware Multilingual Translation System



Enterprise-focused multilingual translation pipeline. Translates text or documents between 30 languages, then automatically verifies its own translation quality via back-translation and retries with a different decoding strategy if confidence is low.



How it works

Translate the input with NLLB-200 — Meta's open-source "No Language Left Behind" model, trained to translate directly between 200 languages without routing through English first.

Translate the result back to the source language (this is the back-translation check — translating A → B → A and seeing how close the result is to the original, a common way to sanity-check a translation without a human reviewer).

Compare the original and the round-trip translation using LaBSE ("Language-agnostic BERT Sentence Embedding") — a model that converts a sentence into a vector so two sentences can be compared for meaning, not just matching words. This gives a semantic similarity score rather than a word-overlap score.

If similarity is below 0.85, retry with a different decoding strategy — the method the model uses to pick output words. It starts with beam search (keeps several likely translations in parallel and picks the best), then tries sampling (introduces controlled randomness to escape a bad translation path), then a wider beam search. Up to 3 attempts, keeping the best-scoring result.

Cache high-quality translations in SQLite (a lightweight, file-based database — no server setup required) so repeated segments are instant on future runs.

Why NLLB + LaBSE, not COMET-Kiwi?



An earlier version of this project used COMET-Kiwi (a popular reference-free translation-quality model) for quality scoring. It was dropped for two reasons:



It's a gated Hugging Face model — requires a Hugging Face account, logging in, and manually accepting a license before it will download. Not something a recruiter running this repo should have to do.

Fragile dependency chain — comet pulls in pytorch\_lightning → torchmetrics, which depends on pkg\_resources, a module setuptools has been removing. This broke on a clean install with no warning until runtime.



LaBSE is open-access, has no license gate, and does the same job (measuring whether two texts mean the same thing) with far fewer moving parts.



Setup

bash

git clone https://github.com/supriyajose1999/context-aware-multilingual-translation-system.git

cd context-aware-multilingual-translation-system

pip install -r requirements.txt

python app.py



Open the local URL Gradio prints in your terminal (usually http://127.0.0.1:7860).



Set GRADIO\_SHARE=true to get a public share link (off by default). No API keys or Hugging Face login required — both models are open-access and download automatically on first use.



Features

Translation — text or PDF/DOCX/TXT upload, with back-translation shown alongside quality/consistency/combined scores

Translation Memory — a searchable cache of past translations, so previously translated content doesn't need to be reprocessed

Glossary — pin domain-specific terms (brand names, technical vocabulary) to a fixed translation so the model doesn't paraphrase them inconsistently

Analytics — usage stats and quality trends over time

Tech stack

Python — core language

PyTorch — deep learning framework the translation and embedding models run on

Hugging Face Transformers — library used to load and run NLLB-200

Sentence-Transformers (LaBSE) — library used to generate the sentence embeddings for quality scoring

Gradio — Python library for building the web UI without writing frontend code

SQLite — local database for translation memory and caching

Evaluation



analysis/evaluation.ipynb (a Jupyter Notebook) benchmarks the pipeline on a small multilingual test set and analyzes the pipeline's own logged usage history. Run it after using the app for a while to get real, measured numbers.



License



MIT

