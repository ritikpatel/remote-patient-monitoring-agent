"""Builds the LangGraph agent graph (PROJECT_PLAN.md section 10).

    DiseaseContext -> VitalsMonitor -> LabInterpreter -> RiskScorer
      -> ContextRetriever -> EscalationDecider -> CarePlanner -> Summarizer

Section 10's original six nodes are unchanged in order and role. Two were added when
the platform became disease-aware, and each sits where it does for a reason:

* **DiseaseContext first.** Three downstream nodes need the diagnosis before they can
  do their job -- ContextRetriever scopes retrieval by `hadm_id` and queries on
  clinical vocabulary rather than score arithmetic, and CarePlanner cannot recommend
  anything for "a deteriorating patient" in the abstract. Putting it anywhere later
  would mean at least one node running disease-blind.
* **CarePlanner after EscalationDecider, before Summarizer.** It must be after the
  escalation decision because it is gated on it (no plan for a patient the policy did
  not escalate), and before the summary because the summary should be able to mention
  that a plan exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

from langgraph.graph import END, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nodes import (  # noqa: E402
    Dependencies,
    audited,
    care_planner,
    context_retriever,
    disease_context,
    escalation_decider,
    lab_interpreter,
    risk_scorer,
    summarizer,
    vitals_monitor,
)
from state import AgentState  # noqa: E402


def build_graph(deps: Dependencies):
    g = StateGraph(AgentState)

    g.add_node("DiseaseContext", audited("DiseaseContext", deps, disease_context(deps)))
    g.add_node("VitalsMonitor", audited("VitalsMonitor", deps, vitals_monitor(deps)))
    g.add_node("LabInterpreter", audited("LabInterpreter", deps, lab_interpreter(deps)))
    g.add_node("RiskScorer", audited("RiskScorer", deps, risk_scorer(deps)))
    g.add_node("ContextRetriever", audited("ContextRetriever", deps, context_retriever(deps)))
    g.add_node("EscalationDecider", audited("EscalationDecider", deps, escalation_decider(deps)))
    g.add_node("CarePlanner", audited("CarePlanner", deps, care_planner(deps)))
    g.add_node("Summarizer", audited("Summarizer", deps, summarizer(deps)))

    g.set_entry_point("DiseaseContext")
    g.add_edge("DiseaseContext", "VitalsMonitor")
    g.add_edge("VitalsMonitor", "LabInterpreter")
    g.add_edge("LabInterpreter", "RiskScorer")
    g.add_edge("RiskScorer", "ContextRetriever")
    g.add_edge("ContextRetriever", "EscalationDecider")
    g.add_edge("EscalationDecider", "CarePlanner")
    g.add_edge("CarePlanner", "Summarizer")
    g.add_edge("Summarizer", END)

    return g.compile()
