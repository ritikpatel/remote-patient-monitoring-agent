from __future__ import annotations

from dataclasses import dataclass

from reports.narrative import generate_narrative


def test_generate_narrative_falls_back_honestly_when_no_llm_configured() -> None:
    result = generate_narrative(None, "system", "some structured facts")
    assert result.startswith("[no LLM configured]")
    assert "some structured facts" in result


@dataclass
class _FakeResult:
    text: str


class _FakeLLM:
    name = "fake"
    model = "fake-model"

    def generate(self, system: str, user: str, max_tokens: int) -> _FakeResult:
        return _FakeResult(text=f"  summary of: {user}  ")


def test_generate_narrative_strips_whitespace_from_a_real_backend() -> None:
    result = generate_narrative(_FakeLLM(), "system", "facts")
    assert result == "summary of: facts"
