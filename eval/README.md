# Evaluation and validation framework

PROJECT_PLAN.md section 13: one HTML report across four axes -- prediction,
alerting, latency, and RAG/agent. Every number in it comes from real code
(Phase 4-6's actual services, Phase 5's actual trained model), not a
re-derivation or a mock.

## Running it

```bash
# Axes 1, 2, 4 need only the warehouse + Python:
python eval/run_eval.py --skip-latency

# Axis 3 (latency) additionally needs k6 (brew install k6) and these five
# services running locally:
uvicorn app:app --app-dir services/ingest-gateway --port 8000 &
uvicorn app:app --app-dir services/stream-processor --port 8003 &
uvicorn app:app --app-dir services/risk-engine --port 8001 &
uvicorn app:app --app-dir services/alert-service --port 8005 &
uvicorn app:app --app-dir services/notification-gateway --port 8006 &
python eval/run_eval.py
```

Set `GROQ_API_KEY` (see `.env`) for axis 4's real agent runs; without it, the
agent graph still runs in full (deterministic path unaffected) and the
report says `[no LLM configured]` instead of a fabricated summary. Output
lands in `eval/output/` (gitignored -- regenerable): `report.html` (open it
directly, or serve `eval/output/` with `python -m http.server` -- see the
color-scheme note below for why a plain `file://` open can look broken in
some browsers) and `manual_review_sample.csv`.

## The four axes, and what each one's real measurement actually is

1. **Prediction** (`eval/prediction.py`) -- AUROC/AUPRC with bootstrap CIs
   (reusing Phase 5's own tested `ml/evaluation/metrics.py`) plus two things
   Phase 5's own report didn't compute: a calibration curve and full
   decision-curve analysis (Vickers & Elkin 2006 -- net benefit of acting on
   the model at each threshold, against "treat everyone" and "treat no
   one"). Runs one fresh grouped holdout split rather than repeating Phase
   5's 20-repeat CV -- that number already exists in
   `ml/evaluation/report.md`; this axis needs raw held-out predictions to
   plot from, not another CV summary.
2. **Alerting** (`eval/alerting.py`) -- a real replay of alert-service's
   actual production dedup/escalation code (`AlertStore.raise_alert`, loaded
   directly, not reimplemented) over every ICU-recalibrated-high hour in the
   warehouse, producing a genuine simulated alert history to measure alerts-
   per-patient-day, median lead time to a true event, sensitivity at a fixed
   alert budget, and false-alarm rate by hour of day (E16) against.
3. **Latency** (`eval/latency.py`, `eval/load/ramp.js`) -- k6 driving the
   real ingest-gateway -> stream-processor -> risk-engine -> alert-service
   -> notification-gateway chain, sequentially, per simulated device event.
   Device-count tiers (10x/100x/1000x) are sized from E12's real fitted peak
   arrival rate (33.7 events/patient/hour, computed fresh from
   `simulators/arrival_models.json`, not hard-coded), not the mean.
4. **RAG and agent** (`eval/rag_agent.py`) -- self-retrieval recall@k against
   the real fact-ledger corpus, a numeric-faithfulness check on real agent-
   generated summaries, escalation agreement with the rule-based policy at
   corpus scale, LLM cost per patient-day from real Groq token usage and
   Groq's real published pricing, and a real 50-summary sample prepared for
   the human manual review the plan requires (see below).

## Latency axis: scope, stated plainly

There is no message bus wiring these five services together yet -- Kafka/
EMQX is Phase 8 infrastructure (PROJECT_PLAN.md section 14). `eval/load/
ramp.js` **is** the orchestrator here: it calls each real service's real
endpoint in sequence and times the whole chain, which is a genuine measure
of each hop's real latency, just orchestrated synchronously by a k6 script
instead of an async consumer. One further, specific limitation: risk-engine's
step scores an *existing* warehouse-backed `(stay_id, hour)`
(`eval/load/fixtures.json`, 50 real pairs), not a value freshly derived from
the observation the same iteration just ingested -- that live "ingest one
point, get a fresh NEWS2" path does not exist (risk-engine's `/score` is a
lookup against the Phase 1 pre-computed hourly grid, not a streaming
computation). Device-count tiers are translated into an aggregate arrival
*rate* (k6's `constant-arrival-rate` executor) rather than literally pacing
1000 individual VUs at ~0.009 events/sec each -- the same throughput/latency
measurement, without k6 needing that many real VUs for no added realism.

**Result:** all three tiers (10x/100x/1000x, up to 181 real iterations at
the top tier) passed with zero failed checks and p95 latency of 35-53ms --
comfortably under PROJECT_PLAN.md section 15's 2-second bar. See
`eval/output/report.html` for the exact numbers from the run that produced
it.

## Alerting axis: a real finding, not a bug

Coverage (the fraction of true composite-deterioration events preceded by
any alert at all) came out low (~15%) in the real replay. This is not a
simulation bug: Phase 5 already found that most patients in this cohort
need vasopressor/ventilator support within 1-3h of ICU admission, often
*before* their NEWS2 has climbed to the "high" tier at all. The raw
NEWS2-high alerting rule genuinely has limited lead time for the specific
events this system targets -- which is the real operational argument for
Phase 5's learned model (its sensitivity-at-a-fixed-alert-budget numbers, in
the same report section, are markedly better), not merely a cross-validation
metric improvement.

## RAG/agent axis: what "faithfulness" and "manual review" mean here

`check_faithfulness` extracts every numeric literal an agent-generated
summary states and checks it against every number in the structured facts
the summarizer was actually given. This catches numeric fabrication (the
most dangerous failure mode for a clinical summary) but not qualitative
misstatement -- a real, useful, but deliberately narrow check, named as such
rather than oversold as full faithfulness evaluation. (Building it also
surfaced a real bug: "NEWS2" and "SpO2" contain a literal digit and were
being extracted as the fabricated number "2" before the number-matching
regex was fixed to require a word boundary.)

PROJECT_PLAN.md section 13 also calls for **manual review of 50 summaries**
-- that is a human judgement this script cannot perform. What it does
instead, honestly: runs the real agent graph (real LLM calls, real
retrieval, real escalation decisions) across 50 real `(stay_id, hour)` pairs
sampled across all three risk tiers, and writes every summary plus an
automated faithfulness pre-check to `eval/output/manual_review_sample.csv`
with blank `REVIEW_*` columns for an actual reviewer to fill in. The report
states this plainly rather than presenting the automated pre-check as a
substitute for the review the plan asks for.

## LLM cost: which model, and why that matters

The $/patient-day figure uses Groq's real published rate for
`openai/gpt-oss-120b` ($0.15/1M input, $0.60/1M output tokens -- fetched
2026-09-06 from Groq's own docs), applied at the more expensive output rate
to the *combined* token total as a deliberate upper bound (agent-orchestrator's
audit log records combined tokens, not split). This is the model this
capstone actually ran against, because no `ANTHROPIC_API_KEY` was available
(see `notes_synth/README.md`) -- PROJECT_PLAN.md's own target backend is
claude-sonnet-5, whose real per-token cost is different, and the report
says so rather than implying this number is that model's cost.

## A real bug this axis found: SVG charts invisible on some browsers

The first rendered report opened as a plain `file://` page looked like it
was missing every chart -- the SVG elements were present in the DOM with
correct geometry (confirmed via devtools), but invisible. The cause: the
page had no explicit `color-scheme`, so some browsers auto-dark-mode
plain HTML content, and the hand-rolled charts' hardcoded stroke colors
(tuned for a white background) become invisible against the inverted one.
Fixed with `:root { color-scheme: light }` plus an explicit white
background on `body` and every `svg.chart` -- the report now renders
identically regardless of the viewing browser's dark-mode setting.
