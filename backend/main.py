"""
main.py  —  TB-Guard FastAPI backend
Uses run_council_parallel from council.py for LangChain-orchestrated
parallel agent execution.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rag"))

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

from council import run_council_parallel          # ← updated import
from rag_pipeline import load_vector_store


app = FastAPI(title="TB-Guard API", version="0.2.0")


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class PatientCase(BaseModel):
    patient_id:    Optional[str] = None
    clinical_data: Optional[str] = None
    genomic_data:  Optional[str] = None
    ct_data:       Optional[str] = None
    xray_data:     Optional[str] = None


class TBDepotCase(BaseModel):
    patient_id:       Optional[str] = None
    subject_id:       Optional[str] = None

    clinical_data:    Optional[str] = None
    clinical_notes:   Optional[str] = None
    demographics:     Optional[str] = None
    symptoms:         Optional[str] = None
    treatment_history: Optional[str] = None

    genomic_data:     Optional[str] = None
    wgs_data:         Optional[str] = None
    mutation_report:  Optional[str] = None

    ct_data:          Optional[str] = None
    ct_findings:      Optional[str] = None
    ct_report:        Optional[str] = None

    xray_data:        Optional[str] = None
    xray_findings:    Optional[str] = None
    cxr_report:       Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _first_non_empty(*values: Optional[str]) -> Optional[str]:
    for v in values:
        if v and v.strip():
            return v.strip()
    return None


def map_tbdepot_to_patient_case(tb: TBDepotCase) -> dict:
    """Normalise the flexible TB-DEPOT schema into the internal flat schema."""
    combined_clinical = "\n".join(
        v for v in [tb.demographics, tb.symptoms, tb.treatment_history]
        if v and v.strip()
    ) or None

    return {
        "patient_id":    _first_non_empty(tb.patient_id, tb.subject_id),
        "clinical_data": _first_non_empty(tb.clinical_data, tb.clinical_notes, combined_clinical),
        "genomic_data":  _first_non_empty(tb.genomic_data, tb.wgs_data, tb.mutation_report),
        "ct_data":       _first_non_empty(tb.ct_data, tb.ct_findings, tb.ct_report),
        "xray_data":     _first_non_empty(tb.xray_data, tb.xray_findings, tb.cxr_report),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}


@app.post("/analyze-case")
def analyze_case(case: PatientCase):
    result = run_council_parallel(case.model_dump())
    return {"input": case.model_dump(), "result": result}


@app.post("/analyze-tbdepot-case")
def analyze_tbdepot_case(tb_case: TBDepotCase):
    mapped = map_tbdepot_to_patient_case(tb_case)
    result = run_council_parallel(mapped)
    return {"input": mapped, "result": result}


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL TEST RUN  (python main.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_case = {
        "clinical_data": """
            34-year-old male, immigrant from high-burden TB region.
            Symptoms: 3-month productive cough, night sweats, 8kg weight loss.
            Prior TB treatment completed 2 years ago. HIV negative.
            Sputum smear: AFB positive (3+).
        """,
        "genomic_data": """
            Whole-genome sequencing results:
            rpoB mutation detected (S450L) — rifampicin resistance.
            katG mutation absent. inhA promoter mutation detected — isoniazid resistance.
            No fluoroquinolone resistance markers detected.
        """,
        "ct_data": None,   # CT agent skipped — no data
        "xray_data": """
            PA chest X-ray: Right upper lobe consolidation with early cavitation.
            Bilateral hilar lymphadenopathy. No pleural effusion.
            Consistent with post-primary tuberculosis pattern.
        """,
    }

    run_council_parallel(test_case)