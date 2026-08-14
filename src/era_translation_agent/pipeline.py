"""Core translation pipeline: NLLB-200 for translation, LaBSE for
quality verification via back-translation semantic similarity.

Why not COMET-Kiwi? It's a gated Hugging Face model (requires login +
license acceptance) and pulls in pytorch_lightning -> torchmetrics,
a dependency chain that breaks on current Windows/Python setups
(pkg_resources removal). LaBSE is open, lightweight, and lets us
verify translations without any external auth."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import (
    LANGUAGE_CODES, NLLB_MODEL_NAME, EMBEDDING_MODEL_NAME,
    QUALITY_THRESHOLD, MAX_QUALITY_ITERATIONS, MEMORY_SAVE_THRESHOLD,
)
from .database import TranslationDatabase

logger = logging.getLogger(__name__)

# Decoding strategies tried in order until quality clears the threshold.
_STRATEGIES = [
    {"num_beams": 4},
    {"num_beams": 1, "do_sample": True, "temperature": 0.7, "top_p": 0.9},
    {"num_beams": 8},
]


class TranslationPipeline:
    def __init__(self, db_path: str = "translation_system.db"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.db = TranslationDatabase(db_path)
        self.memory = self.db.load_translation_memory()

        self._tok = self._model = None
        self._embedder = None

    # -- lazy loaders --------------------------------------------------

    def _load_nllb(self):
        if self._model is None:
            logger.info("Loading NLLB-200 ...")
            self._tok = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                NLLB_MODEL_NAME, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
        return self._tok, self._model

    def _load_embedder(self):
        if self._embedder is None:
            logger.info("Loading LaBSE ...")
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        return self._embedder

    # -- core steps -----------------------------------------------------

    def _generate(self, text: str, src_lang: str, tgt_lang: str, **gen_kwargs) -> str:
        tok, model = self._load_nllb()
        tok.src_lang = LANGUAGE_CODES[src_lang]
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        tgt_id = tok.convert_tokens_to_ids(LANGUAGE_CODES[tgt_lang])
        out = model.generate(**inputs, forced_bos_token_id=tgt_id, max_length=512, **gen_kwargs)
        return tok.batch_decode(out, skip_special_tokens=True)[0]

    def score_similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts' LaBSE embeddings, in [0, 1]."""
        embedder = self._load_embedder()
        embs = embedder.encode([text_a, text_b], normalize_embeddings=True)
        return float(max(0.0, min(1.0, embs[0] @ embs[1])))

    # -- orchestration --------------------------------------------------

    def translate_with_consistency_check(self, text: str, src_lang: str, tgt_lang: str) -> dict:
        """Translate, then back-translate and score how semantically close
        the round trip is to the original. Below QUALITY_THRESHOLD, retries
        with a different decoding strategy (up to MAX_QUALITY_ITERATIONS),
        keeping the best-scoring attempt."""
        cache_key = f"{text}_{src_lang}_{tgt_lang}"
        if cache_key in self.memory:
            cached = self.memory[cache_key]
            return {
                "translation": cached, "back_translation": "", "quality": 1.0,
                "consistency": 1.0, "combined": 1.0, "attempts": 0, "cached": True,
            }

        best = None
        attempt = 0
        for attempt, strategy in enumerate(_STRATEGIES[:MAX_QUALITY_ITERATIONS], start=1):
            try:
                translation = self._generate(text, src_lang, tgt_lang, **strategy)
                back = self._generate(translation, tgt_lang, src_lang, num_beams=4)
                consistency = self.score_similarity(text, back)
            except Exception:
                logger.exception("Translation attempt %d failed", attempt)
                continue

            if best is None or consistency > best["consistency"]:
                best = {"translation": translation, "back_translation": back, "consistency": consistency}
            if consistency >= QUALITY_THRESHOLD:
                break

        if best is None:
            return {
                "translation": "", "back_translation": "", "quality": 0.0, "consistency": 0.0,
                "combined": 0.0, "attempts": attempt, "cached": False, "error": "All attempts failed",
            }

        quality = best["consistency"]
        best.update(quality=quality, combined=quality, attempts=attempt, cached=False)

        self.db.save_translation(text, best["translation"], src_lang, tgt_lang, quality, quality)
        self.db.log_translation(text, src_lang, tgt_lang, quality, quality)
        if quality > MEMORY_SAVE_THRESHOLD:
            self.memory[cache_key] = best["translation"]
        return best

    # Alias kept for compatibility with the evaluation notebook / earlier API.
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> dict:
        return self.translate_with_consistency_check(text, src_lang, tgt_lang)
