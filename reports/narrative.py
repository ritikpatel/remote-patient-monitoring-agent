"""Shared LLM narrative generation -- the same discipline agent-orchestrator's
Summarizer node uses (PROJECT_PLAN.md section 10/12: "rendered from the agent
Summarizer"), reused here rather than re-invented, because report narration
is exactly the same shape of call: synthesise already-computed structured
facts into prose, state only what is given, never invent a number.

Not a literal call into agent-orchestrator's graph -- that graph is scoped to
one (stay_id, hour) escalation decision, which doesn't fit a ward-wide shift
handover or a multi-day trajectory. Same backend protocol
(``services/agent-orchestrator/nodes.py``'s ``LLMBackend``, itself matching
``notes_synth.backends.Backend``), same fallback when no LLM is configured,
same "do not invent" system-prompt instruction.
"""

from __future__ import annotations

from typing import Protocol


class LLMBackend(Protocol):
    name: str
    model: str

    def generate(self, system: str, user: str, max_tokens: int): ...  # noqa: D102


ANTI_FABRICATION_INSTRUCTION = (
    "State only what is given in the structured facts below. Do not invent "
    "vitals, labs, scores, patient counts, or timestamps not present in the "
    "input. If information is missing, say it is missing rather than "
    "guessing. Write in plain prose paragraphs only -- no markdown formatting "
    "(no **bold**, no bullet lists, no headers): this becomes the body text "
    "of a PDF section, which renders markdown syntax as literal characters."
)


def generate_narrative(
    llm: LLMBackend | None, system: str, user: str, max_tokens: int = 400
) -> str:
    """Returns the narrative paragraph, or an explicit fallback string (never
    silently empty) when no LLM is configured -- the same "[no LLM
    configured]" prefix agent-orchestrator's summarizer uses, so a report
    generated without an API key is honest about it rather than blank.
    """
    if llm is None:
        return f"[no LLM configured] {user[:300]}"
    result = llm.generate(system, user, max_tokens=max_tokens)
    return result.text.strip()
