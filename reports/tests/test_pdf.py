from __future__ import annotations

from reports.pdf import ReportSection, _sanitize_for_pdf, render_pdf


def test_sanitize_replaces_unicode_punctuation_with_ascii() -> None:
    # Found for real: a non-breaking hyphen from an LLM narrative rendered as
    # a "tofu" box glyph in the exported PDF.
    assert _sanitize_for_pdf("hours 0‑23") == "hours 0-23"
    assert _sanitize_for_pdf("it’s") == "it's"
    assert _sanitize_for_pdf("a — b") == "a -- b"


def test_sanitize_escapes_xml_special_characters() -> None:
    assert _sanitize_for_pdf("A & B < C") == "A &amp; B &lt; C"


def test_sanitize_strips_stray_markdown_bold_and_headers() -> None:
    assert _sanitize_for_pdf("**Summary**") == "Summary"
    assert _sanitize_for_pdf("## Heading\ntext") == "Heading\ntext"


def test_render_pdf_writes_a_real_valid_pdf_file(tmp_path) -> None:
    out = tmp_path / "report.pdf"
    result = render_pdf(
        out,
        title="Test report",
        generated_note="Covers a test window.",
        sections=[ReportSection("Section one", "Body text.\nSecond line.")],
        citations=["F001", "F002"],
    )
    assert result == out
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-")
