"""Renders the single cross-cutting HTML report PROJECT_PLAN.md section 13
requires: one document, four axes (prediction, alerting, latency, RAG/agent),
each backed by the real measurements ``eval/{prediction,alerting,latency,
rag_agent}.py`` produce. Plain HTML + inline SVG, no external JS/CSS
dependency, no network fetch -- runs and views entirely offline. Every chart
here is a small hand-rolled SVG (same approach as ``ui/src/components/
NewsTraceChart.tsx``), not a charting library.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime

import numpy as np
import pandas as pd

HONEST_REPORTING_NOTICE = (
    "This platform is validated on a 100-patient demo subset of MIMIC-IV. Clinical "
    "narrative is LLM-generated from structured data. Wearable deterioration signals "
    "are synthetically morphed from healthy-volunteer recordings. The engineering is "
    "real and the methodology is rigorous; the clinical performance figures below "
    "demonstrate pipeline validity and do not transfer to clinical practice."
)

CSS = """
:root { color-scheme: light; }
html, body { background: #ffffff; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #1a1d23; }
h1 { border-bottom: 3px solid #2b3a55; padding-bottom: .5rem; }
h2 { margin-top: 3rem; border-bottom: 1px solid #ccc; padding-bottom: .3rem; color: #2b3a55; }
h3 { color: #444; }
.notice { background: #fff3cd; border: 1px solid #ffe69c; padding: .75rem 1rem;
          border-radius: 6px; margin: 1rem 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .92rem; }
th, td { border: 1px solid #ddd; padding: .4rem .6rem; text-align: right; }
th { background: #f4f5f7; text-align: right; }
td:first-child, th:first-child { text-align: left; }
.pass { color: #0f6e56; font-weight: 700; }
.fail { color: #a32d2d; font-weight: 700; }
.muted { color: #666; font-size: .9rem; }
.toc { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem 1.5rem; }
.axis-nav a { margin-right: 1rem; }
svg.chart { border: 1px solid #eee; border-radius: 4px; background: #ffffff; }
"""


def _esc(text: object) -> str:
    return html.escape(str(text))


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "&mdash;"
    return f"{x:.{digits}f}"


def _line_chart_svg(
    series: dict[str, list[tuple[float, float]]],
    width: int = 560,
    height: int = 260,
    x_label: str = "",
    y_label: str = "",
    colors: dict[str, str] | None = None,
) -> str:
    """A minimal hand-rolled SVG line chart -- no charting library, same
    approach as ui/src/components/NewsTraceChart.tsx."""
    colors = colors or {}
    palette = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#b83280", "#718096"]
    padding = 40
    all_points = [p for pts in series.values() for p in pts]
    if not all_points:
        return "<p class='muted'>No data.</p>"
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(0, min(ys)), max(ys)
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    def sx(x: float) -> float:
        return padding + (x - x_min) / x_span * (width - 2 * padding)

    def sy(y: float) -> float:
        return height - padding - (y - y_min) / y_span * (height - 2 * padding)

    parts = [
        f"<svg class='chart' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"
    ]
    parts.append(
        f"<line x1='{padding}' y1='{height - padding}' x2='{width - padding}' "
        f"y2='{height - padding}' stroke='#999'/>"
    )
    parts.append(
        f"<line x1='{padding}' y1='{padding}' x2='{padding}' "
        f"y2='{height - padding}' stroke='#999'/>"
    )
    for i, (name, pts) in enumerate(series.items()):
        color = colors.get(name, palette[i % len(palette)])
        path = " ".join(
            f"{'M' if j == 0 else 'L'} {sx(x):.1f} {sy(y):.1f}" for j, (x, y) in enumerate(pts)
        )
        parts.append(f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2'/>")
        lx, ly = pts[-1]
        parts.append(
            f"<text x='{sx(lx) + 4:.1f}' y='{sy(ly):.1f}' font-size='10' "
            f"fill='{color}'>{_esc(name)}</text>"
        )
    parts.append(
        f"<text x='{width / 2}' y='{height - 6}' font-size='11' text-anchor='middle' "
        f"fill='#555'>{_esc(x_label)}</text>"
    )
    parts.append(
        f"<text x='12' y='{height / 2}' font-size='11' fill='#555' "
        f"transform='rotate(-90 12 {height / 2})' text-anchor='middle'>{_esc(y_label)}</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Axis 1 -- Prediction
# --------------------------------------------------------------------------


def render_prediction_section(summary: dict, horizon: int) -> str:
    rows = []
    for name, entry in summary.items():
        auroc, auprc = entry["auroc"], entry["auprc"]
        rows.append(
            f"<tr><td>{_esc(name)}</td>"
            f"<td>{_fmt(auroc.point)} ({_fmt(auroc.lo)}-{_fmt(auroc.hi)})</td>"
            f"<td>{_fmt(auprc.point)} ({_fmt(auprc.lo)}-{_fmt(auprc.hi)})</td>"
            f"<td>{_fmt(entry.get('brier'))}</td></tr>"
        )
    table = (
        "<table><thead><tr><th>Model</th><th>AUROC (95% CI)</th>"
        "<th>AUPRC (95% CI)</th><th>Brier</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )

    calib_series = {}
    for name in ("age_vitals_lr", "logistic_full", "lightgbm"):
        if name not in summary:
            continue
        curve: pd.DataFrame = summary[name]["calibration"]
        calib_series[name] = list(zip(curve.mean_predicted, curve.observed_rate, strict=True))
    calib_series["perfect calibration"] = [(0, 0), (1, 1)]
    calib_svg = _line_chart_svg(
        calib_series, x_label="mean predicted probability", y_label="observed rate"
    )

    dca_series = {}
    if "lightgbm" in summary and "decision_curve" in summary["lightgbm"]:
        dc = summary["lightgbm"]["decision_curve"]
        dca_series["lightgbm"] = list(zip(dc.threshold, dc.net_benefit_model, strict=True))
        dca_series["treat all"] = list(zip(dc.threshold, dc.net_benefit_treat_all, strict=True))
        dca_series["treat none"] = list(zip(dc.threshold, dc.net_benefit_treat_none, strict=True))
    dca_svg = _line_chart_svg(dca_series, x_label="threshold probability", y_label="net benefit")

    return f"""
    <h2 id="prediction">1. Prediction</h2>
    <p>Composite deterioration, {horizon}h horizon, one grouped holdout split
    (Phase 5's own 20-repeat CV numbers are in <code>ml/evaluation/report.md</code>;
    this table and its charts need raw held-out predictions, not just CV summary
    statistics, so it is a fresh single split rather than a re-run of that CV).</p>
    {table}
    <h3>Calibration</h3>
    <p class="muted">Closer to the diagonal is better-calibrated. NEWS2/SOFA are
    ordinal scores, not probabilities, and are excluded here for the same reason
    Phase 5 excluded them from Brier score.</p>
    {calib_svg}
    <div class="notice"><strong>Subgroup performance and the demographic-feature
    question</strong> are audited in Phase 5's own report
    (<code>ml/evaluation/report.md</code>, "Fairness audit"), which is where the model
    lives. Two results worth carrying here: <code>gender</code> was dropped after a
    20-repeat ablation showed it won in only 13/20 repeats, and subgroup estimates
    whose confidence interval is too wide to mean anything are withheld rather than
    published as findings. The well-measured split is time in stay -- held-out AUPRC is
    0.715 in the first six ICU hours and 0.074 after -- so risk-engine declares an
    <code>in_validated_scope</code> flag on every ML prediction rather than letting a
    consumer assume the headline number applies everywhere. The AUROC/AUPRC on this
    page are cohort averages and do not show any of that.</div>

    <h3>Decision-curve analysis (lightgbm)</h3>
    <p class="muted">Net benefit of acting on the model at each threshold probability,
    against "treat everyone" and "treat no one". The model is only useful where its
    curve sits above both baselines.</p>
    {dca_svg}
    """


# --------------------------------------------------------------------------
# Axis 2 -- Alerting
# --------------------------------------------------------------------------


def render_alerting_section(
    alert_history: pd.DataFrame,
    alerts_per_day: float,
    lead_time_result,
    false_alarm_df: pd.DataFrame,
    sensitivity_by_budget: dict[str, dict[float, float]],
) -> str:
    sens_rows = []
    budgets = sorted({b for m in sensitivity_by_budget.values() for b in m})
    header = "".join(f"<th>{int(b * 100)}% budget</th>" for b in budgets)
    for name, by_budget in sensitivity_by_budget.items():
        cells = "".join(f"<td>{_fmt(by_budget.get(b))}</td>" for b in budgets)
        sens_rows.append(f"<tr><td>{_esc(name)}</td>{cells}</tr>")
    sens_table = (
        f"<table><thead><tr><th>Model</th>{header}</tr></thead>"
        f"<tbody>{''.join(sens_rows)}</tbody></table>"
    )

    fa_series = {
        "false-alarm rate": list(
            zip(false_alarm_df.hour_of_day, false_alarm_df.false_alarm_rate, strict=True)
        )
    }
    fa_svg = _line_chart_svg(fa_series, x_label="hour of day", y_label="false-alarm rate")

    lead_text = (
        f"{_fmt(lead_time_result.median_lead_time_h, 1)}h"
        if lead_time_result.median_lead_time_h is not None
        else "n/a"
    )

    return f"""
    <h2 id="alerting">2. Alerting</h2>
    <p>Real replay of alert-service's actual production dedup/escalation code
    (<code>eval/alerting.py</code>) over every ICU-recalibrated-high hour in the
    warehouse -- {len(alert_history)} alerts raised across the cohort.</p>
    <table><tbody>
      <tr><td>Alerts per patient-day</td><td>{_fmt(alerts_per_day, 2)}</td></tr>
      <tr><td>Median lead time to a true composite event
              (among events with a preceding alert)</td><td>{lead_text}</td></tr>
      <tr><td>Events with a preceding alert</td>
          <td>{lead_time_result.n_events_with_a_preceding_alert} / {lead_time_result.n_events_total}
              ({_fmt(lead_time_result.coverage * 100, 1)}%)</td></tr>
    </tbody></table>
    <div class="notice">Coverage this low is a real, notable finding, not a bug: many
    patients' first composite event happens within 1-3h of ICU admission (Phase 5's
    own finding), often before their NEWS2 has climbed to the "high" tier at all --
    the raw NEWS2-high alerting rule has limited lead time for exactly the events this
    system targets, which is the operational motivation for Phase 5's learned model,
    not just a CV-metric improvement.</div>
    <h3>Sensitivity at a fixed alert budget</h3>
    <p class="muted">If only the top X% of patient-hours by score may be alerted on
    (a realistic staffing constraint), what fraction of true events are caught?</p>
    {sens_table}
    <h3>False-alarm rate by hour of day (E16)</h3>
    <p class="muted">An alert is a "true alarm" if that patient's first composite
    event follows within 12h; a flat rate across hours would itself be notable
    given E16's 4-hourly clock finding.</p>
    {fa_svg}
    """


# --------------------------------------------------------------------------
# Axis 3 -- Latency
# --------------------------------------------------------------------------


def render_latency_section(results: list) -> str:
    rows = []
    for r in results:
        if r.error:
            rows.append(
                f"<tr><td>{r.device_count}x</td>"
                f"<td colspan='5' class='fail'>{_esc(r.error)}</td></tr>"
            )
            continue
        verdict = (
            "<span class='pass'>PASS</span>"
            if r.meets_p95_bar
            else "<span class='fail'>FAIL</span>"
        )
        checks_verdict = (
            "<span class='pass'>all passed</span>"
            if r.all_checks_passed
            else f"<span class='fail'>{r.checks_failed} failed</span>"
        )
        rows.append(
            f"<tr><td>{r.device_count}x ({_fmt(r.events_per_sec, 2)} ev/s)</td>"
            f"<td>{_fmt(r.p50_ms, 1)}</td><td>{_fmt(r.p95_ms, 1)}</td><td>{_fmt(r.p99_ms, 1)}</td>"
            f"<td>{r.n_iterations}</td><td>{checks_verdict}</td><td>{verdict}</td></tr>"
        )
    table = (
        "<table><thead><tr><th>Device count</th><th>p50 (ms)</th><th>p95 (ms)</th>"
        "<th>p99 (ms)</th><th>iterations</th><th>checks</th><th>&lt;2s bar</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return f"""
    <h2 id="latency">3. Latency</h2>
    <p>k6 (<code>eval/load/ramp.js</code>) driving the real ingest-gateway to
    stream-processor to risk-engine to alert-service to notification-gateway chain,
    sequentially, per simulated device event -- see <code>eval/README.md</code> for
    exactly what this measures and does not (there is no Kafka bus wiring these
    together yet; Phase 8 infra). Device-count tiers are sized to E12's real fitted
    peak arrival rate (33.7 events/patient/hour), not the mean.</p>
    {table}
    """


# --------------------------------------------------------------------------
# Axis 4 -- RAG and agent
# --------------------------------------------------------------------------


def render_rag_agent_section(
    recall_result,
    faithfulness_rates: list[float],
    agreement_result,
    cost_result,
    review_sheet: pd.DataFrame,
) -> str:
    mean_faithfulness = float(np.mean(faithfulness_rates)) if faithfulness_rates else float("nan")
    review_preview = review_sheet.head(5)[
        ["patient_ref", "hour", "news2", "escalate", "auto_check_faithfulness_rate"]
    ]
    # .to_dict("records"), not .itertuples() -- pandas-stubs infers itertuples()'s
    # per-column type as a huge dtype union that mypy won't narrow to float; a
    # plain dict's Any values sidestep that without a fragile per-field cast.
    preview_rows = "".join(
        f"<tr><td>{_esc(row['patient_ref'])}</td><td>{row['hour']}</td><td>{row['news2']}</td>"
        f"<td>{row['escalate']}</td>"
        f"<td>{_fmt(float(row['auto_check_faithfulness_rate']))}</td></tr>"
        for row in review_preview.to_dict("records")
    )
    return f"""
    <h2 id="rag-agent">4. RAG and agent</h2>
    <table><tbody>
      <tr><td>Retrieval recall@{recall_result.k}
              (self-retrieval, real fact-ledger corpus)</td>
          <td>{recall_result.hits} / {recall_result.n_samples}
              ({_fmt(recall_result.recall * 100, 1)}%)</td></tr>
      <tr><td>Mean faithfulness rate
              (numeric grounding, {len(faithfulness_rates)} real runs)</td>
          <td>{_fmt(mean_faithfulness * 100, 1)}%</td></tr>
      <tr><td>Escalation agreement with the rule-based policy</td>
          <td>{agreement_result.n_agree} / {agreement_result.n}
              ({_fmt(agreement_result.agreement_rate * 100, 1)}%)</td></tr>
      <tr><td>Mean tokens per run</td>
          <td>{_fmt(cost_result.mean_tokens_per_run, 0)}</td></tr>
      <tr><td>LLM cost per run (mean, upper bound)</td>
          <td>${_fmt(cost_result.mean_cost_per_run_usd, 5)}</td></tr>
      <tr><td>Total LLM cost, this corpus (upper bound)</td>
          <td>${_fmt(cost_result.total_cost_usd_upper_bound, 4)}
              ({cost_result.n_runs} runs)</td></tr>
      <tr><td>Projected LLM cost per patient-day</td>
          <td>${_fmt(cost_result.projected_cost_per_patient_day_usd, 4)}</td></tr>
    </tbody></table>
    <div class="notice">Escalation agreement is 100% by construction --
    EscalationDecider computes <code>escalate</code> from
    <code>warehouse.news2.should_escalate</code> (all three NEWS2 limbs: aggregate
    tier, single red non-GCS parameter, and a falling GCS off sedation) before the LLM
    is ever consulted (Phase 4's constraint 2). Reported here at corpus scale to
    confirm that holds in practice, not as a metric the agent could plausibly fail.
    Note that this metric read 100% while the policy was in fact dropping one of
    NEWS2's triggers, because it held its own hand-copied reference implementation of
    the rule -- both copies were wrong in the same direction (finding F1). It now
    imports the predicate rather than restating it.</div>
    <div class="notice">LLM cost uses Groq's real published rate for
    openai/gpt-oss-120b ($0.60/1M tokens, the more expensive output rate, applied to
    the combined input+output total as a deliberate upper bound) -- the model this
    capstone actually ran against given no ANTHROPIC_API_KEY was available.
    PROJECT_PLAN.md's target backend is claude-sonnet-5, whose real cost differs.</div>
    <h3>Manual review sample (preview)</h3>
    <p class="muted">PROJECT_PLAN.md section 13 calls for manual review of 50
    summaries by a human -- that judgement cannot be performed by this script. The
    full sheet (real agent runs, real summaries, an automated numeric-faithfulness
    pre-check, and blank columns for a reviewer) is at
    <code>eval/output/manual_review_sample.csv</code>.</p>
    <table><thead><tr><th>Patient</th><th>Hour</th><th>NEWS2</th><th>Escalate</th>
    <th>Auto faithfulness</th></tr></thead><tbody>{preview_rows}</tbody></table>
    """


# --------------------------------------------------------------------------
# Full document
# --------------------------------------------------------------------------


def render_report(sections: dict[str, str]) -> str:
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    nav = "".join(
        f'<a href="#{key}">{title}</a>'
        for key, title in [
            ("prediction", "1. Prediction"),
            ("alerting", "2. Alerting"),
            ("latency", "3. Latency"),
            ("rag-agent", "4. RAG and agent"),
        ]
    )
    body = "\n".join(sections.values())
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Evaluation and validation framework</title>
<style>{CSS}</style>
</head>
<body>
<h1>Evaluation and validation framework</h1>
<p class="muted">Generated {generated_at} &middot; PROJECT_PLAN.md section 13</p>
<div class="notice">{HONEST_REPORTING_NOTICE}</div>
<div class="toc axis-nav">{nav}</div>
{body}
</body>
</html>
"""
