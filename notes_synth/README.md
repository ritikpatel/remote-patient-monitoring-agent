# Synthetic clinical narrative

PROJECT_PLAN.md section 9. Unblocks RAG summarisation (Phase 4/6) and, as a
by-product, gives Phase 7 a mechanical way to measure summarisation faithfulness
that real MIMIC notes could never provide.

## Pipeline

```
facts.py       -- extract_facts(hadm_id) -> FactSet: every structured fact one
                  admission's notes are allowed to state, each with a stable
                  fact_id and a (table, row_id, column) trace back to its source row
backends.py    -- AnthropicBackend | GroqBackend | OfflineTemplateBackend, one
                  interface: generate(system, user, max_tokens) -> GenerationResult
generate.py    -- ties them together: builds the prompt, calls a backend, forces
                  the watermark, parses citations, checks faithfulness, logs
                  everything, versions the output via DVC
fact_ledger.py -- sentence splitting, citation extraction, and the mechanical
                  faithfulness check generate.py and Phase 7 both use
```

## Model: a deliberate deviation from the plan

PROJECT_PLAN.md specifies claude-sonnet-5. This environment had no
`ANTHROPIC_API_KEY`; the user supplied a `GROQ_API_KEY` instead and asked
explicitly for **openai/gpt-oss-120b via Groq**. `AnthropicBackend` is fully
implemented and is the default (`--backend anthropic`) — swap the key in and it
runs unmodified. The real generation run committed here used `--backend groq`, on
that explicit instruction, not as a silent substitution. See `backends.py`'s module
docstring.

One real quirk found by running it: gpt-oss-120b is a reasoning model whose hidden
"reasoning" tokens count against `max_tokens` — at the API's default reasoning
effort, a moderately-sized prompt returned an **empty completion** because all the
token budget went to invisible reasoning before any visible text was written.
Fixed with `reasoning_effort="low"`.

## The fact ledger mechanism

Every prompt ends with a numbered `FACTS` list; the model is instructed to prefix
each clinical-fact sentence with the fact id(s) it draws on: `[F003] Sentence.`
`fact_ledger.py` splits the note into sentences (on both `. `-boundaries and
markdown line breaks — a real run showed bullet lists need the latter) and checks
every citation against the exact `FactSet` the model was given. A citation to a
fact id that doesn't exist there is caught, not trusted — this is Phase 7's
"faithfulness" metric, computed here as regression-testable infrastructure rather
than left as a manual read.

## Real run committed here

```bash
python notes_synth/generate.py --n 25 --seed 7 --backend groq \
    --max-total-tokens 500000 --max-tokens-per-note 1500
```

25 admissions sampled, 24 produced at least one note (one hit Groq's on-demand-tier
**daily** token cap on all three of its note types — the per-minute limit retries
with backoff automatically; the daily cap does not, and generate.py logs the
failure and moves on rather than losing the rest of the run).

| | |
|---|---|
| Notes generated | 64 (discharge summaries, ICU nursing notes, radiology stubs — the latter two only where facts.py found grounding for them; ECG stubs are no longer produced, the dataset having been removed from the project) |
| Total tokens | 156,756 |
| Approx. cost | $0.043 (Groq pricing, approximate — see backends.py) |
| Mean faithfulness | **1.000** — zero invalid citations across all 64 notes |

Output: `output/notes/*.txt` (DVC-tracked), `output/fact_catalog.parquet` (every
extracted fact, DVC-tracked), `output/fact_ledger.parquet` (every sentence + its
citations, DVC-tracked), `output/generation_log.csv` (git-tracked directly — small,
and useful in history as a record of what was generated and spent).

## Rebuilding the ledger without spending anything

The citation parser will keep improving (see the markdown-bullet fix above, found
by running a real batch). Re-deriving the ledger from already-generated notes costs
nothing:

```bash
python notes_synth/generate.py --rebuild-ledger
```

## Trying it with no API key at all

```bash
python notes_synth/generate.py --n 5 --backend offline
```

`OfflineTemplateBackend` is not an LLM — it concatenates each fact as its own
self-cited line. Its output reads as a fact list, not prose, deliberately: it
exercises every other part of the pipeline (extraction, ledger, watermarking, DVC)
for free, and should never be mistaken for the real generation backends' output.
