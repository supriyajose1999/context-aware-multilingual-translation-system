"""Gradio UI for the ERA Translation Agent."""

import os
import gradio as gr
import pandas as pd

from src.era_translation_agent import TranslationPipeline, LANGUAGE_CODES
from src.era_translation_agent.document_processing import extract_text

pipeline = TranslationPipeline()
LANGS = sorted(LANGUAGE_CODES.keys())

CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
#hero { text-align: center; padding: 8px 0 4px 0; }
#hero h1 {
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    font-size: 2.1em; margin-bottom: 4px;
}
.score-badge {
    border-radius: 12px; padding: 14px; text-align: center; font-weight: 600;
}
.score-badge .label { font-size: 0.8em; opacity: 0.75; font-weight: 400; }
.score-badge .value { font-size: 1.8em; }
.score-good { background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.4); color: #22c55e; }
.score-mid  { background: rgba(234,179,8,0.15); border: 1px solid rgba(234,179,8,0.4); color: #eab308; }
.score-bad  { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); color: #ef4444; }
"""


def score_badge(label: str, value: float) -> str:
    tier = "score-good" if value >= 0.85 else "score-mid" if value >= 0.7 else "score-bad"
    return f'<div class="score-badge {tier}"><div class="label">{label}</div><div class="value">{value:.2f}</div></div>'


# ---------------------------------------------------------------------
# Translation tab
# ---------------------------------------------------------------------

def run_translation(text: str, file, src_lang: str, tgt_lang: str):
    if file is not None:
        try:
            text = extract_text(file.name)
        except Exception as e:
            err = score_badge("Quality", 0) 
            return "", "", err, err, err, f"Could not read file: {e}"

    if not text or not text.strip():
        empty = score_badge("Quality", 0)
        return "", "", empty, empty, empty, "Enter text or upload a PDF/DOCX/TXT file."

    try:
        r = pipeline.translate_with_consistency_check(text, src_lang.lower(), tgt_lang.lower())
    except Exception as e:
        err = score_badge("Quality", 0)
        return "", "", err, err, err, f"Translation failed: {e}"

    if r.get("error"):
        err = score_badge("Quality", 0)
        return "", "", err, err, err, r["error"]

    status = f"✓ Complete — {r['attempts']} attempt(s)" + (" (from cache)" if r["cached"] else "")
    q = score_badge("Quality", r["quality"])
    c = score_badge("Consistency", r["consistency"])
    cm = score_badge("Combined", r["combined"])
    return r["translation"], r["back_translation"], q, c, cm, status


def build_translation_tab():
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Text", lines=6, placeholder="Type text to translate...")
            input_file = gr.File(label="Or upload a document", file_types=[".pdf", ".docx", ".txt"])
            with gr.Row():
                src = gr.Dropdown(LANGS, value="english", label="Source Language")
                tgt = gr.Dropdown(LANGS, value="spanish", label="Target Language")
            translate_btn = gr.Button("✨ Translate", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="Translation", lines=6)
            back_translation = gr.Textbox(label="Back Translation", lines=4)

    with gr.Row():
        quality = gr.HTML(score_badge("Quality", 0))
        consistency = gr.HTML(score_badge("Consistency", 0))
        combined = gr.HTML(score_badge("Combined", 0))

    status = gr.Textbox(label="Status", interactive=False)

    translate_btn.click(
        run_translation,
        inputs=[input_text, input_file, src, tgt],
        outputs=[output_text, back_translation, quality, consistency, combined, status],
    )


# ---------------------------------------------------------------------
# Translation Memory tab
# ---------------------------------------------------------------------

def search_memory(query: str):
    if query.strip():
        return pipeline.db.search_translation_memory(query)
    return pipeline.db.view_translation_memory()


def build_memory_tab():
    gr.Markdown("Cached translations that scored above the save threshold. Repeated segments are served instantly from here.")
    search_box = gr.Textbox(label="Search (leave blank to view recent)")
    refresh_btn = gr.Button("🔍 Search / Refresh")
    table = gr.Dataframe(value=pipeline.db.view_translation_memory())
    refresh_btn.click(search_memory, inputs=search_box, outputs=table)


# ---------------------------------------------------------------------
# Glossary tab
# ---------------------------------------------------------------------

def add_term(term: str, translation: str, src_lang: str, tgt_lang: str):
    if not term.strip() or not translation.strip():
        return pipeline.db.view_glossary(), "Enter both a term and its translation."
    pipeline.db.add_glossary_term(term.strip(), translation.strip(), src_lang.lower(), tgt_lang.lower())
    return pipeline.db.view_glossary(), f"Added: {term} → {translation}"


def build_glossary_tab():
    gr.Markdown("Domain-specific terms that should always translate the same way (e.g. company or contract terminology).")
    with gr.Row():
        term_in = gr.Textbox(label="Term")
        translation_in = gr.Textbox(label="Translation")
        g_src = gr.Dropdown(LANGS, value="english", label="Source Language")
        g_tgt = gr.Dropdown(LANGS, value="spanish", label="Target Language")
    add_btn = gr.Button("➕ Add Term")
    g_status = gr.Textbox(label="Status", interactive=False)
    g_table = gr.Dataframe(value=pipeline.db.view_glossary())
    add_btn.click(add_term, inputs=[term_in, translation_in, g_src, g_tgt], outputs=[g_table, g_status])


# ---------------------------------------------------------------------
# Analytics tab
# ---------------------------------------------------------------------

def load_analytics(days: int):
    report = pipeline.db.generate_analytics_report(days=int(days))
    stats = pipeline.db.get_statistics(num_supported_languages=len(LANGS))
    summary = (
        f"Total translations ({int(days)}d): {report['total_translations']}\n"
        f"Average quality: {report['avg_quality']:.3f}\n"
        f"Average consistency: {report['avg_consistency']:.3f}\n"
        f"Translation memory size: {stats['translation_memory_size']}\n"
        f"Glossary size: {stats['glossary_size']}"
    )
    daily = pd.DataFrame(report["daily_breakdown"])
    return summary, daily


def build_analytics_tab():
    days = gr.Slider(1, 90, value=7, step=1, label="Days")
    refresh_btn = gr.Button("📊 Refresh")
    summary_box = gr.Textbox(label="Summary", lines=6, interactive=False)
    daily_table = gr.Dataframe(label="Daily breakdown")
    refresh_btn.click(load_analytics, inputs=days, outputs=[summary_box, daily_table])


# ---------------------------------------------------------------------

theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="purple")

with gr.Blocks(title="ERA Translation Agent", theme=theme, css=CSS) as demo:
    gr.HTML('<div id="hero"><h1>🌐 ERA Translation Agent</h1></div>')
    gr.Markdown(
        "Enterprise translation with automatic quality-driven retries (NLLB-200), "
        "verified via back-translation semantic similarity (LaBSE) — no gated models, no external API keys. "
        "Accepts typed text or PDF/DOCX/TXT uploads.",
        elem_id="subtitle",
    )
    with gr.Tabs():
        with gr.Tab("🔤 Translation"):
            build_translation_tab()
        with gr.Tab("🗂️ Translation Memory"):
            build_memory_tab()
        with gr.Tab("📖 Glossary"):
            build_glossary_tab()
        with gr.Tab("📊 Analytics"):
            build_analytics_tab()

if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE", "false").lower() == "true")
