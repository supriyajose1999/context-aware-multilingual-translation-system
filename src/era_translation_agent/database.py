"""
SQLite persistence layer for the ERA Translation Agent.

Handles the translation memory (cache of previously-translated,
high-quality segments), the auto-learned glossary, and daily
analytics. Pulled out of the pipeline class so it can be tested and
reasoned about independently.

Security note: every query in this module uses parameterised
placeholders ("?") rather than string interpolation. The original
prototype built one search query with an f-string, which is a
classic SQL-injection vector -- fixed here.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

import pandas as pd

from .config import DEFAULT_DB_PATH, MEMORY_SAVE_THRESHOLD

logger = logging.getLogger(__name__)


class TranslationDatabase:
    """Thin, safe wrapper around the SQLite database used by the pipeline."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._setup_database()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _setup_database(self) -> None:
        """Create tables if they don't already exist."""
        with self._connection() as conn:
            c = conn.cursor()

            c.execute(
                """CREATE TABLE IF NOT EXISTS translation_memory
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      source_text TEXT,
                      target_text TEXT,
                      src_lang TEXT,
                      tgt_lang TEXT,
                      quality_score REAL,
                      consistency_score REAL,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                      UNIQUE(source_text, src_lang, tgt_lang))"""
            )

            c.execute(
                """CREATE TABLE IF NOT EXISTS glossary
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      term TEXT,
                      translation TEXT,
                      src_lang TEXT,
                      tgt_lang TEXT,
                      frequency INTEGER DEFAULT 1,
                      last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                      UNIQUE(term, src_lang, tgt_lang))"""
            )

            c.execute(
                """CREATE TABLE IF NOT EXISTS analytics
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      date DATE UNIQUE,
                      translations_count INTEGER,
                      avg_quality REAL,
                      avg_consistency REAL,
                      languages_used TEXT,
                      total_chars INTEGER)"""
            )

        logger.info("Database initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Translation memory
    # ------------------------------------------------------------------

    def load_translation_memory(self) -> dict[str, str]:
        """Load high-quality cached translations into an in-memory dict."""
        memory: dict[str, str] = {}
        try:
            with self._connection() as conn:
                rows = conn.execute(
                    "SELECT source_text, target_text, src_lang, tgt_lang "
                    "FROM translation_memory WHERE quality_score > ?",
                    (MEMORY_SAVE_THRESHOLD,),
                ).fetchall()
            for source_text, target_text, src_lang, tgt_lang in rows:
                memory[f"{source_text}_{src_lang}_{tgt_lang}"] = target_text
            logger.info("Loaded %d entries from translation memory", len(memory))
        except sqlite3.Error:
            logger.info("Translation memory is empty or unavailable")
        return memory

    def save_translation(
        self,
        source: str,
        translation: str,
        src_lang: str,
        tgt_lang: str,
        quality: float,
        consistency: float,
    ) -> bool:
        """Persist a high-quality translation. Returns True if it was saved."""
        if quality <= MEMORY_SAVE_THRESHOLD or consistency <= MEMORY_SAVE_THRESHOLD:
            return False

        with self._connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO translation_memory
                    (source_text, target_text, src_lang, tgt_lang, quality_score, consistency_score)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                (source, translation, src_lang, tgt_lang, quality, consistency),
            )
        return True

    def view_translation_memory(self, limit: int = 100) -> pd.DataFrame:
        with self._connection() as conn:
            return pd.read_sql_query(
                "SELECT source_text, target_text, src_lang, tgt_lang, "
                "quality_score, timestamp FROM translation_memory "
                "ORDER BY timestamp DESC LIMIT ?",
                conn,
                params=(limit,),
            )

    def search_translation_memory(self, query: str, limit: int = 50) -> pd.DataFrame:
        """Search cached translations. Uses a parameterised LIKE clause
        (the original prototype built this with an f-string, which was
        vulnerable to SQL injection)."""
        pattern = f"%{query}%"
        with self._connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM translation_memory "
                "WHERE source_text LIKE ? OR target_text LIKE ? LIMIT ?",
                conn,
                params=(pattern, pattern, limit),
            )

    # ------------------------------------------------------------------
    # Glossary
    # ------------------------------------------------------------------

    def load_glossary(self) -> dict[str, str]:
        glossary: dict[str, str] = {}
        try:
            with self._connection() as conn:
                rows = conn.execute(
                    "SELECT term, translation, src_lang, tgt_lang FROM glossary"
                ).fetchall()
            for term, translation, src_lang, tgt_lang in rows:
                glossary[f"{term}_{src_lang}_{tgt_lang}"] = translation
            logger.info("Loaded %d terms from glossary", len(glossary))
        except sqlite3.Error:
            logger.info("Glossary is empty or unavailable")
        return glossary

    def upsert_glossary_terms(
        self, term_pairs: list[tuple[str, str]], src_lang: str, tgt_lang: str
    ) -> None:
        """Insert/increment frequency for a batch of (term, translation) pairs."""
        with self._connection() as conn:
            for s_term, t_term in term_pairs:
                existing = conn.execute(
                    "SELECT frequency FROM glossary WHERE term = ? AND src_lang = ? AND tgt_lang = ?",
                    (s_term, src_lang, tgt_lang),
                ).fetchone()
                new_frequency = (existing[0] + 1) if existing else 1
                conn.execute(
                    """INSERT OR REPLACE INTO glossary
                        (term, translation, src_lang, tgt_lang, frequency)
                        VALUES (?, ?, ?, ?, ?)""",
                    (s_term, t_term, src_lang, tgt_lang, new_frequency),
                )

    def add_glossary_term(
        self, term: str, translation: str, src_lang: str, tgt_lang: str
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO glossary (term, translation, src_lang, tgt_lang, frequency)
                    VALUES (?, ?, ?, ?, COALESCE(
                        (SELECT frequency FROM glossary WHERE term = ? AND src_lang = ? AND tgt_lang = ?) + 1, 1))""",
                (term, translation, src_lang, tgt_lang, term, src_lang, tgt_lang),
            )

    def view_glossary(self, limit: int = 100) -> pd.DataFrame:
        with self._connection() as conn:
            return pd.read_sql_query(
                "SELECT term, translation, src_lang, tgt_lang, frequency, last_updated "
                "FROM glossary ORDER BY frequency DESC LIMIT ?",
                conn,
                params=(limit,),
            )

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def log_translation(
        self, text: str, src_lang: str, tgt_lang: str, quality: float, consistency: float
    ) -> None:
        today = datetime.now().date().isoformat()

        with self._connection() as conn:
            result = conn.execute(
                "SELECT translations_count, avg_quality, avg_consistency, total_chars "
                "FROM analytics WHERE date = ?",
                (today,),
            ).fetchone()

            if result:
                old_count, old_quality, old_consistency, old_chars = result
                new_count = old_count + 1
                new_quality = (old_quality * old_count + quality) / new_count
                new_consistency = (old_consistency * old_count + consistency) / new_count
                new_chars = old_chars + len(text)

                conn.execute(
                    """UPDATE analytics SET
                        translations_count = ?, avg_quality = ?,
                        avg_consistency = ?, total_chars = ?
                        WHERE date = ?""",
                    (new_count, new_quality, new_consistency, new_chars, today),
                )
            else:
                conn.execute(
                    """INSERT INTO analytics
                        (date, translations_count, avg_quality, avg_consistency, languages_used, total_chars)
                        VALUES (?, 1, ?, ?, ?, ?)""",
                    (today, quality, consistency, f"{src_lang}-{tgt_lang}", len(text)),
                )

    def generate_analytics_report(self, days: int = 7) -> dict:
        with self._connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM analytics WHERE date >= date('now', ?) ORDER BY date DESC",
                conn,
                params=(f"-{int(days)} days",),
            )

        return {
            "period": f"Last {days} days",
            "total_translations": int(df["translations_count"].sum()) if len(df) else 0,
            "avg_quality": float(df["avg_quality"].mean()) if len(df) else 0.0,
            "avg_consistency": float(df["avg_consistency"].mean()) if len(df) else 0.0,
            "total_characters": int(df["total_chars"].sum()) if len(df) else 0,
            "daily_breakdown": df.to_dict("records") if len(df) else [],
        }

    def get_statistics(self, num_supported_languages: int) -> dict:
        with self._connection() as conn:
            translation_memory_size = conn.execute(
                "SELECT COUNT(*) FROM translation_memory"
            ).fetchone()[0]
            glossary_size = conn.execute("SELECT COUNT(*) FROM glossary").fetchone()[0]
            total_translations = (
                conn.execute("SELECT SUM(translations_count) FROM analytics").fetchone()[0] or 0
            )
            avg_quality = conn.execute("SELECT AVG(avg_quality) FROM analytics").fetchone()[0] or 0

        return {
            "translation_memory_size": translation_memory_size,
            "glossary_size": glossary_size,
            "total_translations": total_translations,
            "avg_quality": avg_quality,
            "supported_languages": num_supported_languages,
        }
