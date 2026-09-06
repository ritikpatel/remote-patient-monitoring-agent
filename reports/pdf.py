"""PDF rendering, generic across all three report types (PROJECT_PLAN.md
section 12: "...exported to PDF"). reportlab, pure Python, no system
dependency (no cairo/pango to install, unlike e.g. weasyprint) -- appropriate
for a report-generation batch job, not a served microservice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# reportlab's Paragraph interprets a subset of HTML-like markup, so raw
# "&"/"<"/">" from an LLM narrative must be escaped or they're misread as
# markup rather than text. Separately, reportlab's default fonts (Type1,
# WinAnsi-encoded) don't cover every Unicode punctuation character an LLM
# tends to reach for -- found for real: a non-breaking hyphen (U+2011) in a
# Groq-generated narrative rendered as a literal "tofu" box ("0<25>23") in
# the exported PDF. Normalised to ASCII equivalents here rather than by
# switching the whole document to an embedded Unicode font, since these are
# report *body text*, not anything requiring exact typographic fidelity.
_UNICODE_PUNCTUATION_TO_ASCII = {
    "‑": "-",  # non-breaking hyphen
    "‐": "-",  # hyphen
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": "--",  # em dash
    "‘": "'",  # left single quote
    "’": "'",  # right single quote
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "…": "...",  # ellipsis
    " ": " ",  # non-breaking space
}


def _sanitize_for_pdf(text: str) -> str:
    for unicode_char, ascii_equivalent in _UNICODE_PUNCTUATION_TO_ASCII.items():
        text = text.replace(unicode_char, ascii_equivalent)
    text = escape(text)
    # Defence in depth against the LLM ignoring the "no markdown" instruction
    # (ANTI_FABRICATION_INSTRUCTION) -- strip stray markdown syntax rather
    # than render it literally.
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text


HONEST_REPORTING_NOTICE = (
    "This platform is validated on a 100-patient demo subset of MIMIC-IV. Clinical "
    "narrative is LLM-generated from structured data. The engineering is real and the "
    "methodology is rigorous; the clinical content below does not represent a real "
    "patient encounter."
)


@dataclass
class ReportSection:
    heading: str
    body: str


def render_pdf(
    output_path: Path,
    title: str,
    generated_note: str,
    sections: list[ReportSection],
    citations: list[str],
) -> Path:
    """``generated_note`` is a one-line "generated as of ..." / "covers hours
    X-Y" provenance string -- every report states exactly what window of
    data it covers, not just its title.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output_path), pagesize=LETTER)
    story = [
        Paragraph(_sanitize_for_pdf(title), styles["Title"]),
        Paragraph(_sanitize_for_pdf(generated_note), styles["Italic"]),
        Spacer(1, 0.15 * inch),
        Paragraph(HONEST_REPORTING_NOTICE, styles["BodyText"]),
        Spacer(1, 0.25 * inch),
    ]
    for section in sections:
        story.append(Paragraph(_sanitize_for_pdf(section.heading), styles["Heading2"]))
        body = _sanitize_for_pdf(section.body).replace("\n", "<br/>")
        story.append(Paragraph(body, styles["BodyText"]))
        story.append(Spacer(1, 0.2 * inch))

    if citations:
        story.append(Paragraph("Fact-ledger citations", styles["Heading2"]))
        story.append(Paragraph(_sanitize_for_pdf(", ".join(citations)), styles["BodyText"]))

    doc.build(story)
    return output_path
