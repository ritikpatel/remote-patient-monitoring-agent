"""Builds the LangGraph agent graph: VitalsMonitor -> LabInterpreter -> RiskScorer
-> ContextRetriever -> EscalationDecider -> Summarizer (PROJECT_PLAN.md section 10).
"""

from __future__ import annotations

import sys
from pathlib import Path

from langgraph.graph import END, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nodes import (  # noqa: E402
    Dependencies,
    audited,
    context_retriever,
    escalation_decider,
    lab_interpreter,
    risk_scorer,
    summarizer,
    vitals_monitor,
)
from state import AgentState  # noqa: E402


def build_graph(deps: Dependencies):
    g = StateGraph(AgentState)

    g.add_node("VitalsMonitor", audited("VitalsMonitor", deps, vitals_monitor(deps)))
    g.add_node("LabInterpreter", audited("LabInterpreter", deps, lab_interpreter(deps)))
    g.add_node("RiskScorer", audited("RiskScorer", deps, risk_scorer(deps)))
    g.add_node("ContextRetriever", audited("ContextRetriever", deps, context_retriever(deps)))
    g.add_node("EscalationDecider", audited("EscalationDecider", deps, escalation_decider(deps)))
    g.add_node("Summarizer", audited("Summarizer", deps, summarizer(deps)))

    g.set_entry_point("VitalsMonitor")
    g.add_edge("VitalsMonitor", "LabInterpreter")
    g.add_edge("LabInterpreter", "RiskScorer")
    g.add_edge("RiskScorer", "ContextRetriever")
    g.add_edge("ContextRetriever", "EscalationDecider")
    g.add_edge("EscalationDecider", "Summarizer")
    g.add_edge("Summarizer", END)

    return g.compile()
