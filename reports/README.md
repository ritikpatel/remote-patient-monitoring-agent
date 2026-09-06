# Automated clinical reports

PROJECT_PLAN.md section 12: "the missing declared output" identified when
auditing v1.0 against the deliverable list -- three report types, all
rendered from the same LLM discipline agent-orchestrator's Summarizer node
uses, with fact-ledger-grounded structured facts, exported to PDF and (where
a single FHIR subject actually exists) a FHIR DocumentReference.

## Running it

```bash
python reports/run_reports.py shift-handover --ward "Medical Intensive Care Unit (MICU)"
python reports/run_reports.py daily-summary --stay-id 30057454 --day 0
python reports/run_reports.py post-discharge-digest --activity STRESS --participant S01
```

Set `GROQ_API_KEY` (see `.env`) for a real LLM narrative; without it, every
report still generates in full with an explicit `[no LLM configured]`
fallback instead of a blank section. PDFs land in `reports/output/`
(gitignored -- regenerable, not hand-authored).

## What "per ward" and "per day" actually mean here

MIMIC-IV's de-identification shifts each patient's calendar dates
independently (a different random offset per patient) but preserves each
event's real hour-of-day and day-of-week -- E16's periodicity finding (care
runs on a 4-hourly clock) depends on exactly that preserved structure. It
also means **there is no shared wall-clock "now" across different patients'
timelines** to snapshot a ward against simultaneously the way a real hospital
system could. Two of the three reports are designed around this reality
rather than glossing over it:

- **Shift handover** (`shift_handover.py`) covers, per ward, each currently-
  monitored patient's own most recent real 00/04/08/12/16/20-aligned 4-hour
  block -- their own last shift boundary, not a synchronised one across the
  ward. The same "latest available data stands in for now" convention
  `risk-engine`'s `/patients` ward view uses (E1: this data is hourly-
  charted, not streamed, so there is no better notion of "now" to begin
  with).
- **Daily summary** (`daily_summary.py`) uses admission-relative days
  (hours 0-23 = day 0, 24-47 = day 1, ...), not wall-clock days -- R1's
  "anchor to hours-since-admission, never stay-relative position" reasoning
  applied to a reporting boundary instead of a modelling one.
- **Post-discharge digest** (`post_discharge_digest.py`) needs no such
  reconciliation because it isn't MIMIC data at all -- see below.

## The post-discharge digest is morphed, and says so first

R7: "never present replayed or morphed data as measured." The wearable
cohort is healthy volunteers with no ICU link (E10), so a genuine week of
post-discharge deterioration cannot come from real data. This digest reuses
`simulators/morphing.py`'s real transforms (HR retargeting, HRV suppression,
synthesised SpO2) conditioned on a synthetic deterioration trajectory,
compressed into that session's actual recorded duration (tens of minutes)
and divided into 7 buckets labelled "day 1".."day 7" to stand in for a week
-- an explicit demo compression, not a claim that a week was actually
recorded. The LLM is instructed to say so in its first sentence
(`DIGEST_ANTI_FABRICATION_SYSTEM`), and the PDF's provenance line says so
before the narrative does.

## FHIR DocumentReference export: only where a single subject genuinely exists

`daily_summary` has an unambiguous `stay_id -> (subject_id, hadm_id)`
lookup, so it exports a real `DocumentReference` via fhir-mapper's own
`document_reference_to_fhir` (the same mapper every other note in this
system uses -- loaded directly, not re-implemented).

`shift_handover` and `post_discharge_digest` deliberately do **not** export
one:

- A ward-wide handover has no single patient subject. FHIR does have ways to
  scope a document to a group (a `Group` or `List` resource as the subject),
  but building and defending a second resource type for one report is out
  of scope here, so this is stated as a limitation rather than forced into a
  bad fit.
- A wearable participant (e.g. "S01") has no MIMIC `subject_id`/`hadm_id` at
  all (E10: "no patient link") -- there is no `Patient`/`Encounter` pair to
  reference.

## Rendering: real bugs an LLM narrative surfaces that hand-written text never would

`pdf.py`'s `_sanitize_for_pdf` exists because of two real problems found by
actually rendering a Groq-generated narrative into a PDF, not by inspection:

- **A non-breaking hyphen (U+2011) rendered as a "tofu" box** --
  reportlab's default fonts (Type1, WinAnsi-encoded) don't cover every
  Unicode punctuation character an LLM reaches for. Common "smart" typography
  (en/em dashes, curly quotes, ellipsis, non-breaking hyphen/space) is
  normalised to ASCII first.
- **The LLM wrote literal markdown** (`**bold**`, `## Heading`) even before
  being told not to, which reportlab's `Paragraph` doesn't interpret --
  it rendered as literal asterisks and hashes in the PDF. Fixed two ways:
  `ANTI_FABRICATION_INSTRUCTION` now explicitly asks for plain prose, and
  `_sanitize_for_pdf` strips stray markdown syntax defensively regardless.

Every report's PDF also opens with the section 17 honest-reporting notice,
not as a footnote.
