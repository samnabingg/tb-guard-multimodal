import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rag"))

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

from council import (
    clinical_data_agent,
    dna_genomic_agent,
    ct_agent,
    xray_agent,
    judge_agent
)
from rag_pipeline import retrieve, load_vector_store


app = FastAPI(title="TB-Guard API", version="0.1.0")


class PatientCase(BaseModel):
    patient_id: Optional[str] = None
    clinical_data: Optional[str] = None
    genomic_data: Optional[str] = None
    ct_data: Optional[str] = None
    xray_data: Optional[str] = None


class TBDepotCase(BaseModel):
    # Keep this flexible because TB-DEPOT exports may vary by column naming.
    patient_id: Optional[str] = None
    subject_id: Optional[str] = None

    clinical_data: Optional[str] = None
    clinical_notes: Optional[str] = None
    demographics: Optional[str] = None
    symptoms: Optional[str] = None
    treatment_history: Optional[str] = None

    genomic_data: Optional[str] = None
    wgs_data: Optional[str] = None
    mutation_report: Optional[str] = None

    ct_data: Optional[str] = None
    ct_findings: Optional[str] = None
    ct_report: Optional[str] = None

    xray_data: Optional[str] = None
    xray_findings: Optional[str] = None
    cxr_report: Optional[str] = None


def _first_non_empty(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def map_tbdepot_to_patient_case(tb_case: TBDepotCase) -> dict:
    # Map likely TB-DEPOT-style columns to the internal schema used by run_council.
    clinical_data = _first_non_empty(
        tb_case.clinical_data,
        tb_case.clinical_notes,
        "\n".join(
            v for v in [tb_case.demographics, tb_case.symptoms, tb_case.treatment_history] if v and v.strip()
        ) or None,
    )
    genomic_data = _first_non_empty(tb_case.genomic_data, tb_case.wgs_data, tb_case.mutation_report)
    ct_data = _first_non_empty(tb_case.ct_data, tb_case.ct_findings, tb_case.ct_report)
    xray_data = _first_non_empty(tb_case.xray_data, tb_case.xray_findings, tb_case.cxr_report)

    return {
        "patient_id": _first_non_empty(tb_case.patient_id, tb_case.subject_id),
        "clinical_data": clinical_data,
        "genomic_data": genomic_data,
        "ct_data": ct_data,
        "xray_data": xray_data,
    }


def run_council(patient_case: dict) -> dict:
    """
    patient_case is a dict with optional keys:
      - clinical_data  (str or None)
      - genomic_data   (str or None)
      - ct_data        (str or None)
      - xray_data      (str or None)
    Only agents with non-None data are run.
    """
    print("\n[TB-GUARD] Council of AI - Starting deliberation...\n")
    vector_store = load_vector_store()

    # ── ORCHESTRATOR: check which modalities are available ──
    agent_conclusions = []

    if patient_case.get("clinical_data"):
        print("[INFO] Clinical Data Agent (GPT-4o) analyzing...")
        agent_conclusions.append(
            clinical_data_agent(patient_case["clinical_data"], vector_store)
        )

    if patient_case.get("genomic_data"):
        print("[INFO] DNA/Genomic Agent (Gemini 2.5 Flash) analyzing...")
        agent_conclusions.append(
            dna_genomic_agent(patient_case["genomic_data"], vector_store)
        )

    if patient_case.get("ct_data"):
        print("[INFO] CT Agent (Llama 3.3 70B) analyzing...")
        agent_conclusions.append(
            ct_agent(patient_case["ct_data"], vector_store)
        )

    if patient_case.get("xray_data"):
        print("[INFO] X-Ray Agent (Qwen3 32B) analyzing...")
        agent_conclusions.append(
            xray_agent(patient_case["xray_data"], vector_store)
        )

    if not agent_conclusions:
        print("[ERROR] No modality data provided. Cannot run council.")
        return {"error": "No modality data provided"}

    print(f"\n[OK] {len(agent_conclusions)} agent(s) produced conclusions.\n")

    # ── JUDGE + CONSENSUS LOOP (max 2 attempts) ──
    MAX_RAG_RETRIES = 2
    extra_rag_context = ""

    for attempt in range(MAX_RAG_RETRIES):
        print(f"[INFO] Judge Agent (GPT-OSS 120B) deliberating... (attempt {attempt + 1})")
        result = judge_agent(agent_conclusions, vector_store, rag_context=extra_rag_context)

        if not result["needs_rag"]:
            # Consensus reached
            print("\n[OK] Consensus reached.\n")
            break
        else:
            # No consensus — fetch more evidence and retry
            print(f"[INFO] No consensus. Judge requested RAG: '{result['rag_query']}'")
            extra_rag_context = "\n\n".join(
                d.page_content for d in retrieve(result["rag_query"], vector_store)
            )
            print("[INFO] Retrieved additional evidence. Retrying...\n")

    print("=" * 60)
    print(result["verdict_text"])
    print("=" * 60)

    return result


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-case")
def analyze_case(case: PatientCase):
    result = run_council(case.model_dump())
    return {"input": case.model_dump(), "result": result}


@app.post("/analyze-tbdepot-case")
def analyze_tbdepot_case(tb_case: TBDepotCase):
    mapped_case = map_tbdepot_to_patient_case(tb_case)
    result = run_council(mapped_case)
    return {"input": mapped_case, "result": result}


# ── TEST RUN ──
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
        "ct_data": None,   # No CT for this case — CT agent will be skipped
        "xray_data": """
            PA chest X-ray: Right upper lobe consolidation with early cavitation.
            Bilateral hilar lymphadenopathy. No pleural effusion.
            Consistent with post-primary tuberculosis pattern.
        """
    }

    run_council(test_case)