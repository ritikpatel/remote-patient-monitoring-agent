"""What actually writes the note text, behind one interface.

Three backends:
  - AnthropicBackend: PROJECT_PLAN.md section 9's default, calls claude-sonnet-5.
    Needs ANTHROPIC_API_KEY.
  - GroqBackend: calls openai/gpt-oss-120b via Groq's OpenAI-compatible API.
    **A deliberate deviation from the plan's claude-sonnet-5 default**, made on the
    user's explicit direction (no ANTHROPIC_API_KEY was available in this
    environment; a GROQ_API_KEY was supplied instead) -- not a silent substitution.
    Needs GROQ_API_KEY.
  - OfflineTemplateBackend: NOT an LLM -- a deterministic template that concatenates
    fact text, each self-citing, into a mechanically well-formed note. Exists so the
    rest of the pipeline (extraction, citation checking, watermarking, DVC) is fully
    exercisable and testable without any API key or network access or cost. Its
    output reads as a fact list, not prose -- that is the point: it should never be
    mistaken for what an LLM backend produces.

A `.env` file at the repo root (gitignored -- never committed) is read for
GROQ_API_KEY / ANTHROPIC_API_KEY if the environment doesn't already have them set,
so a key pasted once doesn't need re-exporting every session.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from notes_synth.facts import FactSet

T = TypeVar("T")

ANTHROPIC_MODEL_ID = "claude-sonnet-5"
GROQ_MODEL_ID = "openai/gpt-oss-120b"

# Approximate, as of this writing -- Sonnet-tier pricing, $/million tokens. Update
# from https://www.anthropic.com/pricing if this drifts; only used to print a rough
# running cost estimate for the --max-total-tokens budget guard, never billed against.
PRICE_PER_MTOK_INPUT = 3.00
PRICE_PER_MTOK_OUTPUT = 15.00

# Groq's published gpt-oss-120b pricing as of this writing -- less certain than the
# Anthropic figures above; treat as a rough order of magnitude, not a bill. Verify at
# https://groq.com/pricing before relying on this for real budget tracking.
GROQ_PRICE_PER_MTOK_INPUT = 0.15
GROQ_PRICE_PER_MTOK_OUTPUT = 0.75


def _load_dotenv(path: Path | None = None) -> None:
    path = path or Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def _retry_on_rate_limit(call: Callable[[], T], max_retries: int = 6, base_delay: float = 2.0) -> T:
    """Both Anthropic's and Groq's (OpenAI-compatible) SDKs raise an exception with
    `status_code == 429` on a rate limit -- found the hard way running a real batch
    against Groq's on-demand tier (TPM limit hit mid-run, no retry, whole batch
    crashed and every note generated so far would have been lost had generate.py's
    caller not also been made to persist partial output). Retries with exponential
    backoff; re-raises anything that isn't a 429, and re-raises the 429 itself once
    retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 -- narrowed to 429 immediately below
            if getattr(exc, "status_code", None) != 429 or attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt)
            print(f"  rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
    raise RuntimeError("unreachable")  # loop always returns or raises


_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "anthropic": (PRICE_PER_MTOK_INPUT, PRICE_PER_MTOK_OUTPUT),
    "groq": (GROQ_PRICE_PER_MTOK_INPUT, GROQ_PRICE_PER_MTOK_OUTPUT),
    "offline": (0.0, 0.0),
}


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    backend: str

    @property
    def approx_cost_usd(self) -> float:
        price_in, price_out = _PRICING_PER_MTOK.get(self.backend, (0.0, 0.0))
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000


class Backend(Protocol):
    name: str

    def generate(self, system: str, user: str, max_tokens: int) -> GenerationResult: ...


@dataclass
class AnthropicBackend:
    name: str = "anthropic"
    model: str = ANTHROPIC_MODEL_ID

    def __post_init__(self) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it to use --backend anthropic, "
                "or use --backend offline for a template-based (non-LLM) dry run."
            )
        import anthropic

        self._client = anthropic.Anthropic()

    def generate(self, system: str, user: str, max_tokens: int) -> GenerationResult:
        resp = _retry_on_rate_limit(
            lambda: self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return GenerationResult(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model,
            backend=self.name,
        )


@dataclass
class GroqBackend:
    """openai/gpt-oss-120b via Groq's OpenAI-compatible chat completions API. See
    the module docstring: this substitutes for PROJECT_PLAN.md section 9's
    claude-sonnet-5 default on the user's explicit direction, not silently.
    """

    name: str = "groq"
    model: str = GROQ_MODEL_ID
    base_url: str = "https://api.groq.com/openai/v1"
    # gpt-oss-120b is a reasoning model: hidden "reasoning" tokens count against
    # max_tokens and, at the default effort, can consume the entire budget before any
    # visible content is written (found by testing: a 1024-token budget on a
    # moderately-sized note prompt returned an EMPTY completion, all 1024 tokens
    # spent on reasoning). "low" keeps reasoning overhead small enough that this
    # project's note lengths reliably fit.
    reasoning_effort: str = "low"

    def __post_init__(self) -> None:
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set. Set it to use --backend groq, "
                "or use --backend offline for a template-based (non-LLM) dry run."
            )
        import openai

        self._client = openai.OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=self.base_url)

    def generate(self, system: str, user: str, max_tokens: int) -> GenerationResult:
        # Suppressed below: openai's `create()` overloads type `model` and
        # `reasoning_effort` as Literals of OpenAI's own model/effort names, which a
        # Groq-hosted model id and a plain str field can never statically satisfy --
        # both are valid at runtime (this is exactly the OpenAI-compatible-API usage
        # the SDK supports), mypy just can't express "any str" against a Literal union.
        resp = _retry_on_rate_limit(
            lambda: self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                max_tokens=max_tokens,
                reasoning_effort=self.reasoning_effort,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return GenerationResult(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
            backend=self.name,
        )


@dataclass
class OfflineTemplateBackend:
    """Deterministic, not an LLM. Renders every fact given in the prompt as its own
    self-cited sentence, in order. Ignores `system` (there is no model to instruct)
    and takes the FactSet directly rather than parsing it back out of the prompt
    text, since it never needs the prose-generation step at all.
    """

    name: str = "offline"
    model: str = "template-v1"

    def generate_from_facts(self, fact_set: FactSet, note_type: str) -> GenerationResult:
        lines = [f"[{f.fact_id}] {f.text}" for f in fact_set.facts]
        text = " ".join(lines)
        # Rough token proxy so --max-total-tokens still means something in offline
        # mode: ~4 chars/token, matching the usual English-text rule of thumb.
        input_tokens = len(text) // 4
        output_tokens = len(text) // 4
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            backend=self.name,
        )

    def generate(self, system: str, user: str, max_tokens: int) -> GenerationResult:
        raise NotImplementedError("OfflineTemplateBackend uses generate_from_facts, not generate")
