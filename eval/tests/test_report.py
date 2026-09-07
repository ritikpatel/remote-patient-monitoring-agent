from __future__ import annotations

from eval.report import _line_chart_svg, render_report


def test_line_chart_svg_handles_empty_series_without_raising() -> None:
    result = _line_chart_svg({})
    assert "No data" in result


def test_line_chart_svg_produces_a_real_svg_with_a_path_per_series() -> None:
    svg = _line_chart_svg({"a": [(0, 0), (1, 1)], "b": [(0, 1), (1, 0)]})
    assert svg.startswith("<svg")
    assert svg.count("<path") == 2


def test_render_report_declares_light_color_scheme() -> None:
    """Found for real: a plain HTML page with no explicit color-scheme gets
    auto-dark-mode-inverted by some browsers, which made the hand-rolled SVG
    charts' hardcoded stroke colors invisible against the changed background.
    """
    doc = render_report({"a": "<p>section</p>"})
    assert "color-scheme: light" in doc


def test_render_report_includes_the_honest_reporting_notice_and_all_sections() -> None:
    doc = render_report({"prediction": "<p>PRED_MARK</p>", "alerting": "<p>ALERT_MARK</p>"})
    assert "100-patient demo subset of MIMIC-IV" in doc
    assert "PRED_MARK" in doc
    assert "ALERT_MARK" in doc
    assert doc.strip().startswith("<!doctype html>")
