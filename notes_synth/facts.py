"""Extract the structured facts one admission's synthetic notes are allowed to state.

PROJECT_PLAN.md section 9: "Source facts: diagnoses_icd x d_icd_diagnoses.long_title,
prescriptions, procedures_icd, microbiologyevents, labevents abnormal flags, services,
transfers, discharge_location, and the Phase 1 severity scores." Every fact extracted
here gets a stable `fact_id`; `generate.py` gives the LLM this list and instructs it to
cite one after every sentence it derives from a fact -- that citation is what
`fact_ledger.py` checks mechanically in Phase 7 (E9's methodology: the missing-notes
weakness becomes the evaluation).

A Fact is deliberately traceable back to a literal row: `table` + `row_id` + `column`
are enough for a human (or a future retrieval-audit script) to look the source value
up again, not just trust the rendered `text`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ECG_RECORD_LIST = (
    REPO_ROOT
    / "data"
    / "raw"
    / "mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1"
    / "record_list.csv"
)

# Labs worth surfacing even when the demo has hundreds of abnormal rows for one
# admission: cap facts per category so the prompt stays a bounded size regardless of
# how event-dense a stay was (E12's admission burst applies to labs too).
MAX_ABNORMAL_LABS = 15
MAX_MEDICATIONS = 20
MAX_MICROBIOLOGY = 10


@dataclass
class Fact:
    fact_id: str
    hadm_id: int
    table: str  # source table, schema-qualified
    row_id: str  # stable identifier for the source row(s) this fact summarizes
    column: str  # which column(s) the value came from
    text: str  # the human-readable sentence a note-writer can cite

    def as_prompt_line(self) -> str:
        return f"[{self.fact_id}] {self.text}"


@dataclass
class _Counter:
    n: int = 0

    def next_id(self) -> str:
        self.n += 1
        return f"F{self.n:03d}"


@dataclass
class FactSet:
    hadm_id: int
    facts: list[Fact] = field(default_factory=list)

    def by_id(self) -> dict[str, Fact]:
        return {f.fact_id: f for f in self.facts}

    def to_prompt_block(self) -> str:
        return "\n".join(f.as_prompt_line() for f in self.facts)


def load_ecg_index(path: Path = ECG_RECORD_LIST) -> pd.DataFrame | None:
    """The MIMIC-IV-ECG demo's record list (subject_id, ecg_time, ...). Loaded once
    by the caller and passed to extract_facts for every admission -- reading a ~100
    row CSV per admission would be wasteful but harmless; passing it in just makes
    the intent (load once, reuse) explicit.
    """
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["ecg_time"])


def extract_facts(
    conn: duckdb.DuckDBPyConnection,
    hadm_id: int,
    ecg_index: pd.DataFrame | None = None,
) -> FactSet:
    counter = _Counter()
    facts: list[Fact] = []

    def add(table: str, row_id: str, column: str, text: str) -> None:
        facts.append(Fact(counter.next_id(), hadm_id, table, row_id, column, text))

    _add_admission_facts(conn, hadm_id, add)
    _add_diagnosis_facts(conn, hadm_id, add)
    _add_procedure_facts(conn, hadm_id, add)
    _add_medication_facts(conn, hadm_id, add)
    _add_microbiology_facts(conn, hadm_id, add)
    _add_abnormal_lab_facts(conn, hadm_id, add)
    _add_service_facts(conn, hadm_id, add)
    _add_transfer_facts(conn, hadm_id, add)
    _add_radiology_order_facts(conn, hadm_id, add)
    _add_severity_score_facts(conn, hadm_id, add)
    if ecg_index is not None:
        _add_ecg_facts(conn, hadm_id, ecg_index, add)

    return FactSet(hadm_id, facts)


AddFn = "callable"  # documentation only; typed loosely below to keep helpers terse


def _add_admission_facts(conn, hadm_id: int, add) -> None:
    row = conn.execute(
        """
        SELECT admission_type, admission_location, discharge_location, race,
               insurance, marital_status, hospital_expire_flag,
               DATE_DIFF('day', admittime, dischtime) AS los_days,
               edregtime IS NOT NULL AS via_ed
        FROM mimiciv_hosp.admissions WHERE hadm_id = ?
        """,
        [hadm_id],
    ).fetchone()
    if row is None:
        return
    adm_type, adm_loc, disch_loc, race, insurance, marital, died, los_days, via_ed = row
    via = "the Emergency Department" if via_ed else adm_loc.lower()
    add(
        "mimiciv_hosp.admissions",
        str(hadm_id),
        "admission_type,admission_location,via_ed",
        f"Admitted as {adm_type.lower()} via {via}.",
    )
    add(
        "mimiciv_hosp.admissions",
        str(hadm_id),
        "los_days",
        f"Length of hospital stay: {los_days} day(s).",
    )
    if died:
        add(
            "mimiciv_hosp.admissions",
            str(hadm_id),
            "hospital_expire_flag,deathtime",
            "Patient died during this hospitalization.",
        )
    else:
        add(
            "mimiciv_hosp.admissions",
            str(hadm_id),
            "discharge_location",
            f"Discharged alive to: {disch_loc.lower() if disch_loc else 'unknown disposition'}.",
        )


def _add_diagnosis_facts(conn, hadm_id: int, add) -> None:
    rows = conn.execute(
        """
        SELECT d.seq_num, dd.long_title
        FROM mimiciv_hosp.diagnoses_icd d
        JOIN mimiciv_hosp.d_icd_diagnoses dd
          ON d.icd_code = dd.icd_code AND d.icd_version = dd.icd_version
        WHERE d.hadm_id = ?
        ORDER BY d.seq_num
        """,
        [hadm_id],
    ).fetchall()
    for seq_num, title in rows:
        rank = "principal diagnosis" if seq_num == 1 else f"diagnosis #{seq_num}"
        add(
            "mimiciv_hosp.diagnoses_icd",
            f"{hadm_id}:{seq_num}",
            "icd_code",
            f"Diagnosis ({rank}): {title}.",
        )


def _add_procedure_facts(conn, hadm_id: int, add) -> None:
    rows = conn.execute(
        """
        SELECT p.seq_num, p.chartdate, dp.long_title
        FROM mimiciv_hosp.procedures_icd p
        JOIN mimiciv_hosp.d_icd_procedures dp
          ON p.icd_code = dp.icd_code AND p.icd_version = dp.icd_version
        WHERE p.hadm_id = ?
        ORDER BY p.seq_num
        """,
        [hadm_id],
    ).fetchall()
    for seq_num, chartdate, title in rows:
        add(
            "mimiciv_hosp.procedures_icd",
            f"{hadm_id}:{seq_num}",
            "icd_code,chartdate",
            f"Procedure performed on {chartdate}: {title}.",
        )


def _add_medication_facts(conn, hadm_id: int, add) -> None:
    rows = conn.execute(
        """
        SELECT drug, mode(route) AS route, count(*) AS n_orders
        FROM mimiciv_hosp.prescriptions
        WHERE hadm_id = ? AND drug IS NOT NULL
        GROUP BY drug
        ORDER BY n_orders DESC
        LIMIT ?
        """,
        [hadm_id, MAX_MEDICATIONS],
    ).fetchall()
    for drug, route, _n_orders in rows:
        route_txt = f" ({route.lower()})" if route else ""
        add(
            "mimiciv_hosp.prescriptions",
            f"{hadm_id}:{drug}",
            "drug,route",
            f"Medication administered{route_txt}: {drug}.",
        )


def _add_microbiology_facts(conn, hadm_id: int, add) -> None:
    rows = conn.execute(
        """
        SELECT microevent_id, chartdate, spec_type_desc, org_name, ab_name, interpretation
        FROM mimiciv_hosp.microbiologyevents
        WHERE hadm_id = ? AND org_name IS NOT NULL
        ORDER BY chartdate
        LIMIT ?
        """,
        [hadm_id, MAX_MICROBIOLOGY],
    ).fetchall()
    for micro_id, chartdate, spec, org, ab, interp in rows:
        detail = f"{org} isolated from {spec.lower()} on {chartdate}."
        if ab and interp:
            detail += f" Sensitivity to {ab}: {interp}."
        add(
            "mimiciv_hosp.microbiologyevents",
            str(micro_id),
            "org_name,ab_name,interpretation",
            detail,
        )


def _add_abnormal_lab_facts(conn, hadm_id: int, add) -> None:
    rows = conn.execute(
        """
        SELECT l.itemid, di.label, l.value, l.valueuom, l.ref_range_lower, l.ref_range_upper,
               l.charttime, l.labevent_id
        FROM mimiciv_hosp.labevents l
        JOIN mimiciv_hosp.d_labitems di ON l.itemid = di.itemid
        WHERE l.hadm_id = ? AND l.flag = 'abnormal'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY l.itemid ORDER BY l.charttime DESC) = 1
        ORDER BY l.charttime DESC
        LIMIT ?
        """,
        [hadm_id, MAX_ABNORMAL_LABS],
    ).fetchall()
    for _itemid, label, value, unit, lo, hi, charttime, labevent_id in rows:
        # MIMIC-IV redacts some free-text/rare values as literal "___" for
        # de-identification -- skip rather than emit a fact that reads as broken.
        if value is None or "_" in str(value):
            continue
        ref = f" (reference {lo}-{hi})" if lo is not None and hi is not None else ""
        text = f"Abnormal lab, most recent: {label} = {value} {unit or ''}{ref}, on {charttime}."
        add(
            "mimiciv_hosp.labevents",
            str(labevent_id),
            "value,flag",
            text.replace("  ", " "),
        )


def _add_service_facts(conn, hadm_id: int, add) -> None:
    rows = conn.execute(
        "SELECT transfertime, curr_service FROM mimiciv_hosp.services "
        "WHERE hadm_id = ? ORDER BY transfertime",
        [hadm_id],
    ).fetchall()
    if not rows:
        return
    services_seq = " -> ".join(s for _, s in rows)
    add(
        "mimiciv_hosp.services",
        f"{hadm_id}:all",
        "curr_service",
        f"Clinical service assignment over the stay: {services_seq}.",
    )


def _add_transfer_facts(conn, hadm_id: int, add) -> None:
    rows = conn.execute(
        """
        SELECT careunit FROM mimiciv_hosp.transfers
        WHERE hadm_id = ? AND eventtype IN ('admit', 'transfer') AND careunit IS NOT NULL
        ORDER BY intime
        """,
        [hadm_id],
    ).fetchall()
    if not rows:
        return
    seen: list[str] = []
    for (careunit,) in rows:
        if not seen or seen[-1] != careunit:
            seen.append(careunit)
    add(
        "mimiciv_hosp.transfers",
        f"{hadm_id}:all",
        "careunit",
        f"Care unit path: {' -> '.join(seen)}.",
    )


def _add_radiology_order_facts(conn, hadm_id: int, add) -> None:
    rows = conn.execute(
        """
        SELECT poe_id, ordertime, order_subtype
        FROM mimiciv_hosp.poe
        WHERE hadm_id = ? AND order_type = 'Radiology'
        ORDER BY ordertime
        LIMIT 10
        """,
        [hadm_id],
    ).fetchall()
    for poe_id, ordertime, subtype in rows:
        add(
            "mimiciv_hosp.poe",
            poe_id,
            "order_type,order_subtype",
            f"Radiology order placed on {ordertime}: {subtype or 'imaging study'}.",
        )


def _add_severity_score_facts(conn, hadm_id: int, add) -> None:
    sapsii = conn.execute(
        "SELECT max(sapsii) FROM mimiciv_derived.sapsii WHERE hadm_id = ?", [hadm_id]
    ).fetchone()[0]
    if sapsii is not None:
        add(
            "mimiciv_derived.sapsii",
            str(hadm_id),
            "sapsii",
            f"Peak SAPS-II severity score: {sapsii}.",
        )

    oasis = conn.execute(
        "SELECT max(oasis) FROM mimiciv_derived.oasis WHERE hadm_id = ?", [hadm_id]
    ).fetchone()[0]
    if oasis is not None:
        add("mimiciv_derived.oasis", str(hadm_id), "oasis", f"Peak OASIS severity score: {oasis}.")

    sirs = conn.execute(
        "SELECT max(sirs) FROM mimiciv_derived.sirs WHERE hadm_id = ?", [hadm_id]
    ).fetchone()[0]
    if sirs is not None:
        add(
            "mimiciv_derived.sirs",
            str(hadm_id),
            "sirs",
            f"Peak SIRS score: {sirs} (of 4 criteria).",
        )

    sofa = conn.execute(
        """
        SELECT max(s.sofa_24hours) FROM mimiciv_derived.sofa s
        JOIN mimiciv_icu.icustays ie ON s.stay_id = ie.stay_id
        WHERE ie.hadm_id = ?
        """,
        [hadm_id],
    ).fetchone()[0]
    if sofa is not None:
        add(
            "mimiciv_derived.sofa",
            str(hadm_id),
            "sofa_24hours",
            f"Peak 24-hour SOFA score: {sofa}.",
        )

    charlson = conn.execute(
        "SELECT charlson_comorbidity_index FROM mimiciv_derived.charlson WHERE hadm_id = ?",
        [hadm_id],
    ).fetchone()
    if charlson is not None:
        add(
            "mimiciv_derived.charlson",
            str(hadm_id),
            "charlson_comorbidity_index",
            f"Charlson comorbidity index: {charlson[0]}.",
        )

    news2 = conn.execute(
        """
        SELECT max(n.news2), max(n.tier_icu = 'high')
        FROM capstone.news2 n
        JOIN mimiciv_icu.icustays ie ON n.stay_id = ie.stay_id
        WHERE ie.hadm_id = ?
        """,
        [hadm_id],
    ).fetchone()
    if news2 and news2[0] is not None:
        max_news2, reached_icu_high = news2
        add(
            "capstone.news2",
            str(hadm_id),
            "news2,tier_icu",
            f"Peak NEWS2 during ICU stay: {int(max_news2)}"
            + (
                " (reached the ICU-recalibrated high-escalation tier)." if reached_icu_high else "."
            ),
        )


def _add_ecg_facts(conn, hadm_id: int, ecg_index: pd.DataFrame, add) -> None:
    """E9: 92/100 clinical-demo patients have a linked 12-lead ECG, but it is
    episodic and unlabelled (no machine_measurements) -- the fact stops at "an ECG
    was recorded then", which is all the structured data actually supports. Rate/
    rhythm/interval facts require the neurokit2 feature extraction in Phase 5.
    """
    admit = conn.execute(
        "SELECT subject_id, admittime, dischtime FROM mimiciv_hosp.admissions WHERE hadm_id = ?",
        [hadm_id],
    ).fetchone()
    if admit is None:
        return
    subject_id, admittime, dischtime = admit
    studies = ecg_index[
        (ecg_index.subject_id == subject_id)
        & (ecg_index.ecg_time >= pd.Timestamp(admittime))
        & (ecg_index.ecg_time <= pd.Timestamp(dischtime))
    ]
    for _, row in studies.iterrows():
        add(
            "mimic-iv-ecg.record_list",
            str(row.study_id),
            "ecg_time",
            f"A 12-lead ECG was recorded on {row.ecg_time} during this admission "
            "(waveform only; no automated measurements in this demo dataset).",
        )
