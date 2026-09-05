"""Generate synthetic clinical notes from structured MIMIC-IV demo facts, with a
fact ledger alongside every one.

PROJECT_PLAN.md section 9. Notes are LLM-generated (claude-sonnet-5) expansions of
exactly the facts notes_synth/facts.py extracted for that admission -- discharge
summaries, ICU nursing progress notes, and ECG/radiology report stubs (the latter
two only where the structured data actually has grounding: a linked ECG study, or a
Radiology-type order in `poe`). Every sentence that states a clinical fact is
prefixed with the fact id(s) it's based on; notes_synth/fact_ledger.py checks those
citations mechanically against the facts the model was actually given.

Controls (PROJECT_PLAN.md section 9, "Controls"):
  - `--max-total-tokens` caps total spend across the run; generation stops (not
    mid-note) once the running total would exceed it.
  - Every call is logged to notes_synth/generation_log.csv (tokens, approx cost,
    faithfulness rate, invalid-citation count) -- capped and logged per run.
  - Every note is watermarked "SYNTHETIC -- generated from MIMIC-IV demo structured
    data" as its first line -- enforced in code, not just requested of the model
    (R7): the header is written by this script regardless of what the model itself
    produced.
  - `--dvc` (default) versions the output via `dvc add` after a successful run.

Usage:
    # no ANTHROPIC_API_KEY needed -- exercises the whole pipeline for real, just not
    # with LLM-quality prose (see notes_synth/backends.py's OfflineTemplateBackend)
    python notes_synth/generate.py --n 5 --backend offline

    # real generation (needs ANTHROPIC_API_KEY)
    python notes_synth/generate.py --n 20 --backend anthropic --max-total-tokens 200000

    # real generation via Groq's openai/gpt-oss-120b (needs GROQ_API_KEY) -- see
    # notes_synth/backends.py's module docstring for why this backend exists
    python notes_synth/generate.py --n 20 --backend groq --max-total-tokens 200000
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notes_synth.backends import (  # noqa: E402
    AnthropicBackend,
    Backend,
    GenerationResult,
    GroqBackend,
    OfflineTemplateBackend,
)
from notes_synth.fact_ledger import (  # noqa: E402
    Sentence,
    check_faithfulness,
    facts_to_dataframe,
    ledger_from_note,
    ledger_to_dataframe,
    write_parquet,
)
from notes_synth.facts import FactSet, extract_facts, load_ecg_index  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
OUT_DIR = REPO_ROOT / "notes_synth" / "output"
NOTES_DIR = OUT_DIR / "notes"
FACT_CATALOG_PATH = OUT_DIR / "fact_catalog.parquet"
FACT_LEDGER_PATH = OUT_DIR / "fact_ledger.parquet"
LOG_PATH = OUT_DIR / "generation_log.csv"

WATERMARK = "SYNTHETIC -- generated from MIMIC-IV demo structured data"

SYSTEM_PROMPT = """You write synthetic clinical documentation for a research pipeline \
that trains and evaluates automated summarisation. You will be given a numbered list \
of FACTS about one hospital admission.

Rules, followed exactly:
1. State ONLY things drawn from the FACTS list below. Never invent a lab value, date, \
medication, diagnosis, or outcome that is not listed.
2. Every sentence that states a clinical fact must be PREFIXED with the fact id(s) it \
is based on: "[F003] Sentence text." A sentence drawing on more than one fact: \
"[F003,F007] Sentence text."
3. When a sentence draws on more than two facts, list every single fact id explicitly, \
comma-separated, inside one set of brackets, e.g. "[F078,F079,F080,F081]". NEVER use a \
range or dash shorthand such as "[F078-F087]" -- it cannot be checked automatically \
and will be treated as uncited.
4. Purely stylistic or transition sentences (e.g. a section header, "Hospital \
course:") do not need a citation -- but keep these to a minimum; most sentences \
should cite something.
5. Do not restate the same fact twice unless clinically natural (e.g. summarising \
then detailing).
6. Write in a professional clinical register appropriate for the requested note \
type. Do not include a greeting, sign-off, or any text about being an AI.
7. Do not include the watermark line yourself -- it is added separately."""

NOTE_TYPE_PROMPTS = {
    "discharge_summary": "Write a discharge summary for this admission.",
    "nursing_progress_note": (
        "Write one daily ICU nursing progress note for this admission, covering "
        "severity scores, care unit, and clinical course."
    ),
    "ecg_report_stub": (
        "Write a brief ECG report stub noting that a 12-lead ECG was recorded "
        "during this admission. Do not invent rate, rhythm, or interval findings "
        "not present in the facts -- state only that a recording exists."
    ),
    "radiology_report_stub": (
        "Write a brief radiology report stub listing the imaging studies ordered "
        "during this admission. Do not invent findings -- state only that the "
        "studies were ordered."
    ),
}


def eligible_note_types(fact_set: FactSet) -> list[str]:
    tables = {f.table for f in fact_set.facts}
    types = ["discharge_summary"]
    if "mimiciv_derived.sofa" in tables or "capstone.news2" in tables:
        types.append("nursing_progress_note")
    if "mimic-iv-ecg.record_list" in tables:
        types.append("ecg_report_stub")
    if "mimiciv_hosp.poe" in tables:
        types.append("radiology_report_stub")
    return types


@dataclass
class LogRow:
    hadm_id: int
    note_type: str
    backend: str
    model: str
    input_tokens: int
    output_tokens: int
    approx_cost_usd: float
    n_facts: int
    n_sentences: int
    n_cited: int
    n_invalid_citations: int
    faithfulness_rate: float
    timestamp: str


def build_note_body(result: GenerationResult) -> str:
    return f"{WATERMARK}\n\n{result.text.strip()}\n"


def generate_one(
    backend: Backend | OfflineTemplateBackend,
    fact_set: FactSet,
    note_type: str,
    max_tokens: int,
) -> GenerationResult:
    if isinstance(backend, OfflineTemplateBackend):
        return backend.generate_from_facts(fact_set, note_type)
    user = f"FACTS:\n{fact_set.to_prompt_block()}\n\n{NOTE_TYPE_PROMPTS[note_type]}"
    return backend.generate(SYSTEM_PROMPT, user, max_tokens)


def run(
    conn: duckdb.DuckDBPyConnection,
    hadm_ids: list[int],
    backend: Backend | OfflineTemplateBackend,
    max_total_tokens: int,
    max_tokens_per_note: int,
    note_types_filter: set[str] | None,
) -> tuple[list[FactSet], list[Sentence], list[LogRow]]:
    ecg_index = load_ecg_index()
    all_fact_sets: list[FactSet] = []
    all_sentences: list[Sentence] = []
    log_rows: list[LogRow] = []
    total_tokens = 0

    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    for hadm_id in hadm_ids:
        fact_set = extract_facts(conn, hadm_id, ecg_index)
        all_fact_sets.append(fact_set)
        types = eligible_note_types(fact_set)
        if note_types_filter:
            types = [t for t in types if t in note_types_filter]

        for note_type in types:
            if total_tokens >= max_total_tokens:
                print(
                    f"Stopping: max-total-tokens ({max_total_tokens:,}) reached "
                    f"after {len(log_rows)} notes.",
                    file=sys.stderr,
                )
                return all_fact_sets, all_sentences, log_rows

            try:
                result = generate_one(backend, fact_set, note_type, max_tokens_per_note)
            except (
                Exception
            ) as exc:  # noqa: BLE001 -- one bad call must not lose the rest of the run
                print(f"  {hadm_id}_{note_type}: FAILED ({exc}) -- skipping", file=sys.stderr)
                continue
            total_tokens += result.input_tokens + result.output_tokens

            note_id = f"{hadm_id}_{note_type}"
            body = build_note_body(result)
            (NOTES_DIR / f"{note_id}.txt").write_text(body)

            sentences = ledger_from_note(note_id, hadm_id, note_type, result.text)
            all_sentences.extend(sentences)
            report = check_faithfulness(sentences, fact_set)

            log_rows.append(
                LogRow(
                    hadm_id=hadm_id,
                    note_type=note_type,
                    backend=result.backend,
                    model=result.model,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    approx_cost_usd=result.approx_cost_usd,
                    n_facts=len(fact_set.facts),
                    n_sentences=report.n_sentences,
                    n_cited=report.n_cited,
                    n_invalid_citations=report.n_invalid_citations,
                    faithfulness_rate=report.faithfulness_rate,
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )
            total_note_tokens = result.input_tokens + result.output_tokens
            print(
                f"  {note_id:<40} {result.backend:<9} tok={total_note_tokens:>6} "
                f"faithfulness={report.faithfulness_rate:.2f}"
                + (
                    f" INVALID_CITATIONS={report.n_invalid_citations}"
                    if report.n_invalid_citations
                    else ""
                )
            )

    return all_fact_sets, all_sentences, log_rows


def rebuild_ledger_from_existing_notes(
    conn: duckdb.DuckDBPyConnection, notes_dir: Path = NOTES_DIR
) -> tuple[list[FactSet], list[Sentence]]:
    """Re-derive the fact catalog and sentence ledger from already-generated note
    files, without calling any LLM. Exists because the citation parser (sentence
    splitting, citation extraction) is expected to improve over time -- e.g. the
    markdown-bullet splitting fix this module picked up after a real run exposed it
    -- and re-running the API for every admission just to re-check citations already
    on disk would be wasteful (and, against a rate-limited backend, often
    impossible: this is exactly the situation that motivated writing it).
    """
    ecg_index = load_ecg_index()
    fact_sets_by_hadm: dict[int, FactSet] = {}
    all_sentences: list[Sentence] = []
    for path in sorted(notes_dir.glob("*.txt")):
        hadm_id_str, _, note_type = path.stem.partition("_")
        hadm_id = int(hadm_id_str)
        if hadm_id not in fact_sets_by_hadm:
            fact_sets_by_hadm[hadm_id] = extract_facts(conn, hadm_id, ecg_index)
        body = path.read_text()
        text = body.split("\n\n", 1)[1] if body.startswith(WATERMARK) else body
        all_sentences.extend(ledger_from_note(path.stem, hadm_id, note_type, text))
    return list(fact_sets_by_hadm.values()), all_sentences


def refresh_log_faithfulness(
    log_path: Path, sentences: list[Sentence], fact_sets: list[FactSet]
) -> None:
    """Update the faithfulness columns of an existing generation_log.csv in place,
    after rebuild_ledger_from_existing_notes reprocesses with a newer parser --
    token counts and cost are untouched (no new API calls happened).
    """
    if not log_path.exists():
        return
    fact_sets_by_hadm = {fs.hadm_id: fs for fs in fact_sets}
    by_note_id: dict[str, list] = {}
    for s in sentences:
        by_note_id.setdefault(s.note_id, []).append(s)

    rows = list(csv.DictReader(open(log_path)))
    for row in rows:
        note_id = f"{row['hadm_id']}_{row['note_type']}"
        note_sentences = by_note_id.get(note_id, [])
        fs = fact_sets_by_hadm.get(int(row["hadm_id"]))
        if not note_sentences or fs is None:
            continue
        report = check_faithfulness(note_sentences, fs)
        row["n_sentences"] = report.n_sentences
        row["n_cited"] = report.n_cited
        row["n_invalid_citations"] = report.n_invalid_citations
        row["faithfulness_rate"] = report.faithfulness_rate

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    fact_sets: list[FactSet], sentences: list[Sentence], log_rows: list[LogRow]
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet(facts_to_dataframe(fact_sets), FACT_CATALOG_PATH)
    write_parquet(ledger_to_dataframe(sentences), FACT_LEDGER_PATH)
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(log_rows[0]).keys()) if log_rows else [])
        if log_rows:
            writer.writeheader()
            for row in log_rows:
                writer.writerow(asdict(row))


def _dvc_executable() -> str:
    # `dvc` is a project dependency (uv-managed venv), not necessarily on the caller's
    # PATH -- look next to the running interpreter first (.venv/bin/dvc) before
    # falling back to PATH.
    venv_dvc = Path(sys.executable).parent / "dvc"
    return str(venv_dvc) if venv_dvc.exists() else "dvc"


def dvc_add(paths: list[Path]) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if not existing:
        return
    try:
        subprocess.run(
            [_dvc_executable(), "add", *existing],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"DVC-tracked: {', '.join(existing)}")
    except FileNotFoundError:
        print(
            "dvc not found -- skipping versioning (run `uv run dvc add ...` manually).",
            file=sys.stderr,
        )
    except subprocess.CalledProcessError as exc:
        print(f"dvc add failed: {exc.stderr}", file=sys.stderr)


def sample_hadm_ids(conn: duckdb.DuckDBPyConnection, n: int, seed: int) -> list[int]:
    """A reproducible random sample, done in Python rather than SQL -- DuckDB's
    `USING SAMPLE` clause takes a literal count/percentage, not a bound parameter.
    """
    all_ids = [r[0] for r in conn.execute("SELECT hadm_id FROM mimiciv_hosp.admissions").fetchall()]
    rng = random.Random(seed)
    return sorted(rng.sample(all_ids, min(n, len(all_ids))))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--hadm-ids", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=None, help="sample N admissions (ordered by hadm_id)")
    ap.add_argument("--all", action="store_true", help="process every admission")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["anthropic", "groq", "offline"], default="anthropic")
    ap.add_argument("--note-types", nargs="*", choices=list(NOTE_TYPE_PROMPTS), default=None)
    ap.add_argument("--max-total-tokens", type=int, default=200_000)
    ap.add_argument("--max-tokens-per-note", type=int, default=1024)
    ap.add_argument("--dvc", action="store_true", default=True)
    ap.add_argument("--no-dvc", dest="dvc", action="store_false")
    ap.add_argument(
        "--rebuild-ledger",
        action="store_true",
        help="re-derive the fact catalog/ledger from existing notes/*.txt -- no LLM calls, no cost",
    )
    args = ap.parse_args()

    if args.rebuild_ledger:
        conn = duckdb.connect(str(args.db), read_only=True)
        fact_sets, sentences = rebuild_ledger_from_existing_notes(conn)
        conn.close()
        if not fact_sets:
            print(f"No notes found under {NOTES_DIR}.", file=sys.stderr)
            return 1
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_parquet(facts_to_dataframe(fact_sets), FACT_CATALOG_PATH)
        write_parquet(ledger_to_dataframe(sentences), FACT_LEDGER_PATH)
        refresh_log_faithfulness(LOG_PATH, sentences, fact_sets)
        print(f"Rebuilt ledger for {len(fact_sets)} admissions from existing notes in {NOTES_DIR}.")
        if args.dvc:
            dvc_add([NOTES_DIR, FACT_CATALOG_PATH, FACT_LEDGER_PATH])
        return 0

    if not any([args.hadm_ids, args.n, args.all]):
        print("ERROR: pass one of --hadm-ids, --n, --all, or --rebuild-ledger.", file=sys.stderr)
        return 1

    backend: Backend | OfflineTemplateBackend
    try:
        if args.backend == "anthropic":
            backend = AnthropicBackend()
        elif args.backend == "groq":
            backend = GroqBackend()
        else:
            backend = OfflineTemplateBackend()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    conn = duckdb.connect(str(args.db), read_only=True)
    if args.hadm_ids:
        hadm_ids = args.hadm_ids
    elif args.all:
        hadm_ids = [
            r[0]
            for r in conn.execute(
                "SELECT hadm_id FROM mimiciv_hosp.admissions ORDER BY hadm_id"
            ).fetchall()
        ]
    else:
        hadm_ids = sample_hadm_ids(conn, args.n, args.seed)

    print(f"Generating for {len(hadm_ids)} admission(s), backend={backend.name}")
    print(f"Model: {backend.model}")
    note_types_filter = set(args.note_types) if args.note_types else None
    fact_sets, sentences, log_rows = run(
        conn, hadm_ids, backend, args.max_total_tokens, args.max_tokens_per_note, note_types_filter
    )
    conn.close()

    if not log_rows:
        print("No notes generated.", file=sys.stderr)
        return 1

    write_outputs(fact_sets, sentences, log_rows)

    total_tokens = sum(r.input_tokens + r.output_tokens for r in log_rows)
    total_cost = sum(r.approx_cost_usd for r in log_rows)
    total_invalid = sum(r.n_invalid_citations for r in log_rows)
    mean_faithfulness = sum(r.faithfulness_rate for r in log_rows) / len(log_rows)
    print(
        f"\n{len(log_rows)} notes across {len(fact_sets)} admissions. "
        f"{total_tokens:,} tokens (~${total_cost:.4f}). "
        f"Mean faithfulness {mean_faithfulness:.3f}. Invalid citations: {total_invalid}."
    )
    print(f"Wrote {NOTES_DIR}, {FACT_CATALOG_PATH.name}, {FACT_LEDGER_PATH.name}, {LOG_PATH.name}")

    if args.dvc:
        dvc_add([NOTES_DIR, FACT_CATALOG_PATH, FACT_LEDGER_PATH])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
