"""
main.py  —  TB-Guard FastAPI backend  v0.4.0

Data sources (honest):
  ✅ clinical_symptoms.csv    — patient symptoms + demographics
  ✅ DST_SAMPLES.parquet      — CRyPTIC/WHO drug resistance per isolate
  ✅ PREDICTIONS.parquet      — per-drug S/R predictions (pivoted)
  ❌ ct_data.parquet          — excluded (random PubMed cases, not TB CT data)
  ❌ clinical_data.csv        — excluded (CDC aggregate surveillance, not patients)
  ❌ xray_metadata.csv        — excluded (just filenames, no clinical data)

TB Portals live integration:
  Set TBPORTALS_EMAIL + TBPORTALS_SECRET in .env to use real API.
  Automatically falls back to local data when credentials absent.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rag"))

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from council import run_council_parallel
from rag_pipeline import load_vector_store
from data_loader import build_joined_records

# TB Portals live integration (optional)
try:
    from tbdepot.client import get_patient_list, get_full_record
    from tbdepot.mapper import map_full_record_to_patient_case, map_patient_case_to_summary
    from tbdepot.auth import credentials_available
    TBDEPOT_AVAILABLE = True
except ImportError:
    TBDEPOT_AVAILABLE = False
    def credentials_available(): return False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TB-Guard API", version="0.4.0")


def _parse_allowed_origins() -> list[str]:
    raw = os.getenv("FRONTEND_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or [
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5173", "http://127.0.0.1:5173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class PatientCase(BaseModel):
    patient_id:    Optional[str] = None
    clinical_data: Optional[str] = None
    genomic_data:  Optional[str] = None
    ct_data:       Optional[str] = None
    xray_data:     Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    from pathlib import Path
    xray_available = (Path("data/shenzhen_xray_stats.csv").exists() and
                      Path("data/shenzhen_clinical.csv").exists())
    return {
        "status": "ok",
        "version": "0.4.0",
        "tbportals_live": credentials_available(),
        "data_sources": {
            "clinical_symptoms": True,
            "dst_samples":       True,
            "predictions":       True,
            "ct_data":           xray_available,
            "xray_data":         xray_available,
        },
    }


@app.get("/tbdepot/patients")
async def get_tbdepot_patients(
    limit: int = Query(default=50, ge=1, le=200),
    skip:  int = Query(default=0,  ge=0),
):
    """
    Returns paginated patient records.
    Uses live TB Portals API if credentials configured,
    otherwise uses local clinical_symptoms + DST data.
    """
    if TBDEPOT_AVAILABLE and credentials_available():
        try:
            raw_page, total, source = await get_patient_list(limit=limit, skip=skip)
            patients = [map_patient_case_to_summary(pc, source=source) for pc in raw_page]
            return {"total": total, "skip": skip, "limit": limit,
                    "source": source, "patients": patients}
        except Exception as exc:
            logger.warning("TB Portals API failed, using local data: %s", exc)

    all_records = build_joined_records()
    total = len(all_records)
    page  = all_records[skip: skip + limit]

    patients = [{
        "condition_id":       r["condition_id"],
        "patient_id":         r["patient_id"],
        "name":               r["name"],
        "age":                r.get("age"),
        "sex":                r["gender"],
        "country":            r["country"],
        "type_of_resistance": r["type_of_resistance"],
        "case_definition":    "Unknown",
        "hiv_status":         "Unknown",
        "has_genomics":       bool(r["drug_profile"]),
        "has_cxr":            bool(r.get("has_cxr", r.get("has_xray"))),
        "has_ct":             bool(r.get("has_ct")),
        "has_dst":            True,
        "data_source":        "local",
        "judge_verdict":      r["judge_verdict"],
        "n_dst_datasets":     r["n_dst_datasets"],
    } for r in page]

    return {"total": total, "skip": skip, "limit": limit,
            "source": "local", "patients": patients}


@app.get("/tbdepot/patients/{condition_id}")
async def get_tbdepot_patient(condition_id: str):
    if TBDEPOT_AVAILABLE and credentials_available():
        try:
            full_record, source = await get_full_record(condition_id)
            if full_record:
                return {"condition_id": condition_id, "source": source, "data": full_record}
        except Exception as exc:
            logger.warning("TB Portals patient fetch failed: %s", exc)

    all_records = build_joined_records()
    record = next((r for r in all_records if r["condition_id"] == condition_id), None)
    if not record:
        raise HTTPException(status_code=404, detail=f"Patient '{condition_id}' not found.")
    return {"condition_id": condition_id, "source": "local", "data": record}


@app.post("/tbdepot/analyze/{condition_id}")
async def analyze_tbdepot_patient(condition_id: str):
    """
    Fetch the full record for condition_id then run council agents.
    Only runs agents for modalities with real data (clinical + genomic).
    """
    if TBDEPOT_AVAILABLE and credentials_available():
        try:
            full_record, source = await get_full_record(condition_id)
            if full_record:
                patient_case = map_full_record_to_patient_case(full_record)
                result = await asyncio.to_thread(run_council_parallel, patient_case)
                return {"condition_id": condition_id, "source": source,
                        "patient_case": patient_case, "result": result}
        except Exception as exc:
            logger.warning("TB Portals analyze failed, using local: %s", exc)

    all_records = build_joined_records()
    record = next((r for r in all_records if r["condition_id"] == condition_id), None)
    if not record:
        raise HTTPException(status_code=404, detail=f"Patient '{condition_id}' not found.")

    patient_case = {
        "patient_id":    record["patient_id"],
        "clinical_data": record["clinical_data"],
        "genomic_data":  record["genomic_data"],
        "ct_data":       record["ct_data"],
        "xray_data":     record["xray_data"],
    }

    available = [k for k in ["clinical_data","genomic_data","ct_data","xray_data"]
                 if patient_case.get(k)]
    logger.info("Running council for %s — modalities: %s", condition_id, available)

    result = await asyncio.to_thread(run_council_parallel, patient_case)
    return {"condition_id": condition_id, "source": "local",
            "patient_case": patient_case, "result": result}


@app.post("/analyze-case")
async def analyze_case(case: PatientCase):
    result = await asyncio.to_thread(run_council_parallel, case.model_dump())
    return {"input": case.model_dump(), "result": result}


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL TEST RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    records = build_joined_records()
    print(f"\nLoaded {len(records)} patient records")
    r = records[0]
    for k in ["patient_id","name","gender","country","type_of_resistance",
              "clinical_data","genomic_data"]:
        print(f"  {k}: {str(r.get(k,''))[:120]}")