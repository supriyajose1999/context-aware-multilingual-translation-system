"""Configuration constants for the ERA Translation Agent."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------
# Minimum semantic round-trip similarity (0-1, cosine) a translation must
# reach before it's accepted. Below this, the pipeline retries with a
# different decoding strategy up to MAX_QUALITY_ITERATIONS times.
QUALITY_THRESHOLD: float = 0.85
MAX_QUALITY_ITERATIONS: int = 3
MEMORY_SAVE_THRESHOLD: float = 0.70

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Both are open-access (no Hugging Face login/license gate required) --
# unlike COMET-Kiwi, which is gated and pulled in a fragile dependency
# chain (pytorch_lightning -> torchmetrics -> pkg_resources) that breaks
# on recent Windows/Python setups. NLLB alone handles translation; LaBSE
# (a multilingual sentence embedding model) is used to verify quality via
# back-translation similarity instead.
NLLB_MODEL_NAME: str = "facebook/nllb-200-distilled-600M"
EMBEDDING_MODEL_NAME: str = "sentence-transformers/LaBSE"

# ---------------------------------------------------------------------------
# Language support (FLORES-200 codes used by NLLB)
# ---------------------------------------------------------------------------

LANGUAGE_CODES: dict[str, str] = {
    "english": "eng_Latn", "spanish": "spa_Latn", "french": "fra_Latn",
    "german": "deu_Latn", "chinese": "zho_Hans", "japanese": "jpn_Jpan",
    "korean": "kor_Hang", "arabic": "arb_Arab", "hindi": "hin_Deva",
    "russian": "rus_Cyrl", "vietnamese": "vie_Latn", "portuguese": "por_Latn",
    "italian": "ita_Latn", "dutch": "nld_Latn", "polish": "pol_Latn",
    "turkish": "tur_Latn", "swedish": "swe_Latn", "thai": "tha_Thai",
    "indonesian": "ind_Latn", "malay": "zsm_Latn", "tagalog": "tgl_Latn",
    "bengali": "ben_Beng", "urdu": "urd_Arab", "persian": "pes_Arab",
    "hebrew": "heb_Hebr", "greek": "ell_Grek", "czech": "ces_Latn",
    "romanian": "ron_Latn", "hungarian": "hun_Latn", "ukrainian": "ukr_Cyrl",
}

DEFAULT_DB_PATH: str = "translation_system.db"
