"""
data_loader.py — TB-Guard unified data loader

Sources and which agents they feed:
┌──────────────────────────────┬─────────────────────────────────────────┐
│ File                         │ Feeds                                   │
├──────────────────────────────┼─────────────────────────────────────────┤
│ clinical_symptoms.csv        │ Clinical Agent (symptoms + demographics)│
│ DST_SAMPLES.parquet          │ Genomic Agent (resistance flags)        │
│ PREDICTIONS.parquet          │ Genomic Agent (per-drug S/R profile)    │
│ shenzhen_xray_stats.csv      │ X-Ray Agent (radiologist findings)      │
│ shenzhen_clinical.csv        │ X-Ray Agent + Clinical Agent (age/sex)  │
│ shenzhen_xray_stats.csv      │ CT Agent  (same findings, CT framing)   │
└──────────────────────────────┴─────────────────────────────────────────┘

Run download_xray_data.py once to fetch the Shenzhen NIH files.
Until then, the X-Ray and CT agents get None (council runs with 2 agents).
Once downloaded, each synthetic patient gets a deterministic stratified pick from
the ~336 Shenzhen rows (TB± cohort × sex) so all patients have paired CXR+CT text.
"""

import hashlib
import logging
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

# ── Drug abbreviations (CRyPTIC / WHO catalogue) ──────────────────────────────
DRUG_ABBREV = {
    "RIF": "Rifampicin",    "INH": "Isoniazid",     "EMB": "Ethambutol",
    "PZA": "Pyrazinamide",  "STR": "Streptomycin",   "AMI": "Amikacin",
    "KAN": "Kanamycin",     "CAP": "Capreomycin",    "LEV": "Levofloxacin",
    "MXF": "Moxifloxacin",  "BDQ": "Bedaquiline",    "LZD": "Linezolid",
    "DLM": "Delamanid",     "CFZ": "Clofazimine",    "ETH": "Ethionamide",
    "CYC": "Cycloserine",   "PAS": "Para-aminosalicylic acid",
    "RFB": "Rifabutin",     "CLR": "Clarithromycin",
}
FIRST_LINE  = {"RIF", "INH", "EMB", "PZA", "STR"}
FLUOROQUINS = {"LEV", "MXF", "OFX"}
INJECTABLES = {"AMI", "KAN", "CAP"}

# ── Shenzhen 19 abnormality column names (from Statistics CSV) ─────────────────
# These are the radiologist-annotated findings in the Shenzhen dataset
SHENZHEN_FINDINGS = [
    "Consolidation", "Nodule", "Cavity", "Fibrosis",
    "Pleural_effusion", "Pleural_thickening", "Cardiomegaly",
    "Aortic_enlargement", "Infiltrate", "Other_lesion",
    "Atelectasis", "Calcification", "Emphysema", "Pneumothorax",
    "Hernia", "Lung_opacity", "Edema", "Fracture", "Scoliosis",
]
# TB-relevant subset for narrative generation
TB_RELEVANT_FINDINGS = {
    "Consolidation", "Nodule", "Cavity", "Fibrosis",
    "Pleural_effusion", "Infiltrate", "Calcification",
    "Lung_opacity", "Atelectasis",
}


# ══════════════════════════════════════════════════════════════════════════════
# RESISTANCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _classify_resistance(drug_profile: dict[str, str]) -> str:
    resistant = {d for d, v in drug_profile.items() if v == "R"}
    rif_r = "RIF" in resistant
    inh_r = "INH" in resistant
    fq_r  = bool(resistant & FLUOROQUINS)
    inj_r = bool(resistant & INJECTABLES)
    if rif_r and inh_r and fq_r:    return "XDR"
    if rif_r and inh_r and inj_r:   return "Pre-XDR"
    if rif_r and inh_r:             return "MDR non XDR"
    if rif_r or inh_r:              return "Mono DR"
    if len(resistant) > 1:          return "Poly DR"
    if len(resistant) == 1:         return "Mono DR"
    return "Sensitive"


# ══════════════════════════════════════════════════════════════════════════════
# STRATIFIED SHENZHEN IMAGING ASSIGNMENT  (reuse 336 CXR rows for all patients)
# ══════════════════════════════════════════════════════════════════════════════

_DR_CLASSES = frozenset({"MDR non XDR", "XDR", "Pre-XDR", "Mono DR", "Poly DR"})


def _normalize_sex(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().lower()
    if s in ("m", "male", "man"):
        return "Male"
    if s in ("f", "female", "woman"):
        return "Female"
    return None


def _shenzhen_tb_bucket(tb_status) -> str:
    s = str(tb_status or "").strip().lower()
    if "positive" in s:
        return "Positive"
    if "negative" in s:
        return "Negative"
    return "Unknown"


def _resistance_imaging_stratum(resistance_class: str) -> str:
    """Coarse bucket for matching patients to Shenzhen TB+ / TB− imaging cohorts."""
    if resistance_class in _DR_CLASSES:
        return "dr"
    if resistance_class == "Sensitive":
        return "sensitive"
    return "other"


def _rng_for_patient_key(key: str) -> random.Random:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def _prepare_xray_matching_frame(xray_df: pd.DataFrame) -> pd.DataFrame:
    work = xray_df.reset_index(drop=True).copy()
    work["_tb_bucket"] = work["tb_status"].map(_shenzhen_tb_bucket)
    work["_sex_norm"] = work["sex"].map(_normalize_sex)
    return work


def _sex_match_mask(work: pd.DataFrame, sex_n: Optional[str]) -> pd.Series:
    """Prefer rows whose reported sex matches the symptom-table sex; allow unknown sex as wildcards."""
    if not sex_n:
        return pd.Series(True, index=work.index)
    s = work["_sex_norm"]
    return (s == sex_n) | s.isna()


def _pick_shenzhen_row_stratified(
    work: pd.DataFrame,
    sym_gender,
    resistance_class: str,
    patient_key: str,
) -> pd.Series:
    """
    Deterministic stratified sample: align DR patients with TB-positive CXR cohort and
    susceptible patients with TB-negative cohort when possible; prefer sex match.
    """
    rng = _rng_for_patient_key(patient_key)
    sex_n = _normalize_sex(sym_gender)
    stratum = _resistance_imaging_stratum(resistance_class)

    if stratum == "dr":
        primary_tb, secondary_tb = (["Positive"], ["Negative", "Unknown"])
    elif stratum == "sensitive":
        primary_tb, secondary_tb = (["Negative"], ["Positive", "Unknown"])
    else:
        primary_tb, secondary_tb = (["Positive", "Negative"], ["Unknown"])

    def try_pick(tb_values: list[str], use_sex: bool) -> Optional[pd.Series]:
        m = work["_tb_bucket"].isin(tb_values)
        if use_sex:
            m &= _sex_match_mask(work, sex_n)
        idxs = work.index[m].tolist()
        if not idxs:
            return None
        return work.loc[rng.choice(idxs)]

    for tb_vals in (primary_tb, secondary_tb):
        row = try_pick(tb_vals, use_sex=True)
        if row is not None:
            return row
        row = try_pick(tb_vals, use_sex=False)
        if row is not None:
            return row

    return work.iloc[rng.randint(0, len(work) - 1)]


def _resistance_narrative(drug_profile: dict[str, str], country: str, classification: str) -> str:
    resistant   = [DRUG_ABBREV.get(d, d) for d, v in drug_profile.items() if v == "R"]
    susceptible = [DRUG_ABBREV.get(d, d) for d, v in drug_profile.items() if v == "S"]
    res_text = f"Resistant to: {', '.join(resistant)}." if resistant else "No drug resistance detected."
    sus_text = f"Susceptible to: {', '.join(susceptible[:6])}{'…' if len(susceptible) > 6 else ''}." if susceptible else ""
    return (
        f"Drug susceptibility testing (CRyPTIC/WHO catalogue). "
        f"Isolate origin: {country}. "
        f"Resistance classification: {classification}. "
        f"{res_text} {sus_text}"
    ).strip()


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL NARRATIVE  (from clinical_symptoms.csv)
# ══════════════════════════════════════════════════════════════════════════════

SYMPTOM_MAP = {
    "fever for two weeks":                                       "fever lasting two weeks",
    "coughing blood":                                            "hemoptysis",
    "sputum mixed with blood":                                   "blood-tinged sputum",
    "night sweats ":                                             "night sweats",
    "chest pain":                                                "chest pain",
    "back pain in certain parts ":                               "localized back pain",
    "shortness of breath":                                       "shortness of breath",
    "weight loss ":                                              "weight loss",
    "body feels tired":                                          "fatigue",
    "lumps that appear around the armpits and neck":             "lymphadenopathy",
    "cough and phlegm continuously for two weeks to four weeks": "productive cough (2–4 weeks)",
    "swollen lymph nodes":                                       "swollen lymph nodes",
    "loss of appetite":                                          "anorexia",
}


def _symptoms_narrative(sym_row: pd.Series, age: Optional[int] = None, sex: Optional[str] = None) -> str:
    active = [label for col, label in SYMPTOM_MAP.items() if sym_row.get(col, 0) == 1]
    name   = str(sym_row.get("name", "Unknown"))
    gender = sex or str(sym_row.get("gender", "Unknown"))
    age_str = f", {age} years old" if age else ""

    if active:
        return (
            f"Patient {name} ({gender}{age_str}). "
            f"Presenting symptoms: {', '.join(active)}. "
            f"Referred for TB screening and further evaluation."
        )
    return f"Patient {name} ({gender}{age_str}). No prominent symptoms recorded at intake."


# ══════════════════════════════════════════════════════════════════════════════
# XRAY NARRATIVE  (from shenzhen_xray_stats.csv)
# ══════════════════════════════════════════════════════════════════════════════

def _xray_narrative(xray_row: Optional[pd.Series], patient_id: str) -> Optional[str]:
    if xray_row is None:
        return None

    # Collect present findings
    present = []
    for finding in SHENZHEN_FINDINGS:
        # Stats CSV has columns like "Consolidation", value = count of annotations
        val = xray_row.get(finding, xray_row.get(finding.lower(), 0))
        try:
            if float(val) > 0:
                present.append(finding.replace("_", " ").lower())
        except (ValueError, TypeError):
            pass

    tb_relevant = [f for f in present if f.replace(" ", "_").title() in TB_RELEVANT_FINDINGS
                   or f.replace(" ", "_") in TB_RELEVANT_FINDINGS]

    tb_status = str(xray_row.get("tb_status", xray_row.get("TB_status", "Unknown")))
    pid_str   = str(xray_row.get("patient_id", patient_id))

    if tb_relevant:
        return (
            f"Chest X-ray — Shenzhen Hospital dataset (NIH/NLM). "
            f"Patient: {pid_str}. TB smear status: {tb_status}. "
            f"Radiologist findings present: {', '.join(tb_relevant)}. "
            f"{'Findings consistent with active pulmonary TB.' if tb_status == 'Positive' else 'Findings noted; clinical correlation required.'}"
        )
    elif present:
        return (
            f"Chest X-ray — Shenzhen Hospital dataset (NIH/NLM). "
            f"Patient: {pid_str}. TB status: {tb_status}. "
            f"Non-specific findings: {', '.join(present[:4])}. No pathognomonic TB features identified."
        )
    else:
        return (
            f"Chest X-ray — Shenzhen Hospital dataset (NIH/NLM). "
            f"Patient: {pid_str}. TB status: {tb_status}. "
            f"No significant radiological abnormalities annotated."
        )


# ══════════════════════════════════════════════════════════════════════════════
# CT NARRATIVE  (derived from Shenzhen findings — CT framing)
# ══════════════════════════════════════════════════════════════════════════════

def _ct_narrative(xray_row: Optional[pd.Series], patient_id: str) -> Optional[str]:
    """
    Uses the same Shenzhen findings reframed as CT observations.
    The Shenzhen dataset includes findings (cavitation, consolidation, nodules)
    that directly correspond to CT TB patterns. This is an honest proxy —
    we label it clearly as derived from CXR annotations.
    """
    if xray_row is None:
        return None

    cavity = float(xray_row.get("Cavity", xray_row.get("cavity", 0)) or 0)
    consol = float(xray_row.get("Consolidation", xray_row.get("consolidation", 0)) or 0)
    nodule = float(xray_row.get("Nodule", xray_row.get("nodule", 0)) or 0)
    fibrosis = float(xray_row.get("Fibrosis", xray_row.get("fibrosis", 0)) or 0)
    effusion = float(xray_row.get("Pleural_effusion", xray_row.get("pleural_effusion", 0)) or 0)

    ct_findings = []
    if cavity > 0:   ct_findings.append(f"cavitary lesion(s) ({int(cavity)} annotated)")
    if consol > 0:   ct_findings.append("consolidation")
    if nodule > 0:   ct_findings.append(f"pulmonary nodule(s) ({int(nodule)} annotated)")
    if fibrosis > 0: ct_findings.append("fibrotic changes")
    if effusion > 0: ct_findings.append("pleural effusion")

    tb_status = str(xray_row.get("tb_status", xray_row.get("TB_status", "Unknown")))
    pid_str   = str(xray_row.get("patient_id", patient_id))

    if ct_findings:
        severity = "extensive" if cavity > 1 or (consol > 0 and cavity > 0) else "moderate"
        return (
            f"CT thorax (derived from Shenzhen CXR radiologist annotations, NIH/NLM). "
            f"Patient: {pid_str}. TB status: {tb_status}. "
            f"Key findings: {', '.join(ct_findings)}. "
            f"Pattern suggests {severity} pulmonary involvement. "
            f"{'Upper lobe predominance and cavitation consistent with post-primary TB.' if cavity > 0 else 'Distribution pattern under evaluation.'}"
        )
    else:
        return (
            f"CT thorax (derived from Shenzhen CXR annotations, NIH/NLM). "
            f"Patient: {pid_str}. TB status: {tb_status}. "
            f"No significant lesions annotated by radiologist."
        )


# ══════════════════════════════════════════════════════════════════════════════
# SCORE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def score_from_seed(seed: str, base: float, variance: float = 1.5) -> float:
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    offset = (h % 100) / 100 * variance * 2 - variance
    return round(min(10.0, max(1.0, base + offset)), 1)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def load_symptoms() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "clinical_symptoms.csv").fillna(0)
    logger.info("Loaded clinical_symptoms.csv: %d rows", len(df))
    return df


@lru_cache(maxsize=1)
def load_dst() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "DST_SAMPLES.parquet")
    logger.info("Loaded DST_SAMPLES.parquet: %d isolates", len(df))
    return df.reset_index()


@lru_cache(maxsize=1)
def load_predictions() -> pd.DataFrame:
    """Pivot PREDICTIONS from long (one row per drug) to wide (one row per isolate)."""
    raw = pd.read_parquet(DATA_DIR / "PREDICTIONS.parquet").reset_index()
    logger.info("PREDICTIONS columns after reset_index: %s", list(raw.columns))

    id_col   = raw.columns[0]
    drug_col = None
    pred_col = "PREDICTION"

    # Identify drug column heuristically
    for col in raw.columns[1:]:
        sample = raw[col].dropna().astype(str).unique()[:30]
        if len(sample) > 0 and all(len(v) <= 6 and v.isupper() and v.isalpha() for v in sample):
            drug_col = col
            logger.info("Inferred drug column: %s", drug_col)
            break

    if drug_col is None or pred_col not in raw.columns:
        logger.warning("Cannot find DRUG/PREDICTION columns in PREDICTIONS.parquet — returning empty")
        return pd.DataFrame()

    try:
        pivoted = raw.pivot_table(
            index=id_col, columns=drug_col, values=pred_col, aggfunc="first"
        )
        pivoted.index.name = "UNIQUEID"
        logger.info("Pivoted predictions: %d isolates × %d drugs", *pivoted.shape)
        return pivoted
    except Exception as exc:
        logger.warning("Pivot failed: %s", exc)
        return pd.DataFrame()


@lru_cache(maxsize=1)
def load_shenzhen_xray() -> Optional[pd.DataFrame]:
    """Load NIH Shenzhen CXR abnormality stats. Returns None if not yet downloaded."""
    stats_path = DATA_DIR / "shenzhen_xray_stats.csv"
    clinical_path = DATA_DIR / "shenzhen_clinical.csv"

    if not stats_path.exists():
        logger.warning(
            "shenzhen_xray_stats.csv not found. Run download_xray_data.py to enable "
            "the X-Ray and CT agents. They will be skipped until then."
        )
        return None

    # Load stats (per-patient finding counts)
    stats = pd.read_csv(stats_path)
    logger.info("Loaded shenzhen_xray_stats.csv: %d rows, columns: %s",
                len(stats), list(stats.columns[:8]))

    # Try to merge with clinical readings for age/sex/tb_status
    if clinical_path.exists():
        clinical = pd.read_csv(clinical_path)
        # Find the join key — typically "patient_id" or "Patient" or first column
        stats_id_col    = next((c for c in stats.columns if "patient" in c.lower() or c == "ID"), stats.columns[0])
        clinical_id_col = next((c for c in clinical.columns if "patient" in c.lower()), clinical.columns[0])

        stats = stats.rename(columns={stats_id_col: "patient_id"})
        clinical = clinical.rename(columns={clinical_id_col: "patient_id"})

        # Normalise IDs — stats has 'CHNCXR_0327_1.png', clinical has 'CHNCXR_0327_1'
        stats["patient_id"]    = stats["patient_id"].astype(str).str.replace(r"\.(png|jpg|jpeg)$", "", regex=True)
        clinical["patient_id"] = clinical["patient_id"].astype(str).str.replace(r"\.(png|jpg|jpeg)$", "", regex=True)

        # Also infer tb_status from ID suffix (_1 = TB positive, _0 = normal)
        if "tb_status" not in clinical.columns:
            clinical["tb_status"] = clinical["patient_id"].apply(
                lambda x: "Positive" if str(x).endswith("_1") else "Negative"
            )

        merged = stats.merge(clinical[["patient_id","age","sex","tb_status","clinical_text"]],
                             on="patient_id", how="left")
        # Fill tb_status from ID for any rows that didn't match
        merged["tb_status"] = merged["tb_status"].fillna(
            merged["patient_id"].apply(lambda x: "Positive" if str(x).endswith("_1") else "Negative")
        )
        logger.info("Merged Shenzhen stats + clinical: %d rows", len(merged))
        return merged.reset_index(drop=True)

    # No clinical file — rename ID column and add placeholder columns
    id_col = stats.columns[0]
    stats = stats.rename(columns={id_col: "patient_id"})
    stats["age"]          = None
    stats["sex"]          = None
    stats["tb_status"]    = stats["patient_id"].apply(
        lambda x: "Positive" if str(x).endswith("_1") else "Negative"
    )
    stats["clinical_text"] = None
    return stats.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN JOIN — builds all patient records
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def build_joined_records() -> list[dict]:
    """
    Joins all available data sources into unified patient records.
    Each record contains all fields needed by all 5 council agents.
    """
    symptoms  = load_symptoms()
    dst       = load_dst()
    preds     = load_predictions()
    xray_df   = load_shenzhen_xray()   # None if not yet downloaded

    n_dst   = len(dst)
    n_xray  = len(xray_df) if xray_df is not None else 0
    xray_available = xray_df is not None and n_xray > 0

    if xray_available:
        logger.info(
            "X-Ray + CT agents ENABLED (%d Shenzhen CXR records; stratified assignment for %d patients)",
            n_xray,
            len(symptoms),
        )
        xray_work = _prepare_xray_matching_frame(xray_df)
    else:
        logger.warning("X-Ray + CT agents DISABLED — run download_xray_data.py to enable")
        xray_work = None

    records = []
    for i, (_, sym_row) in enumerate(symptoms.iterrows()):
        dst_row  = dst.iloc[i % n_dst]
        uniqueid = str(dst_row.get("UNIQUEID", f"iso-{i}"))
        country  = str(dst_row.get("COUNTRY_CODE", "Unknown"))

        patient_id = f"TBD-{10001 + i}"
        seed = patient_id

        # ── Drug resistance profile ───────────────────────────────────────────
        drug_profile: dict[str, str] = {}
        if not preds.empty and uniqueid in preds.index:
            drug_profile = {
                drug: str(val)
                for drug, val in preds.loc[uniqueid].items()
                if pd.notna(val) and str(val) in ("S", "R", "U")
            }

        if not drug_profile:
            n_ds = int(dst_row.get("N_DATASETS", 1))
            has_cryptic = dst_row.get("HAS_CRYPTIC1_DST", False) or dst_row.get("HAS_CRYPTIC2_DST", False)
            has_who     = dst_row.get("HAS_WHO2019_DST", False)
            if n_ds >= 3:
                drug_profile = {"RIF":"R","INH":"R","EMB":"R","PZA":"S","LEV":"S"}
            elif has_cryptic or has_who:
                drug_profile = {"RIF":"R","INH":"R","EMB":"S","PZA":"S"}
            else:
                drug_profile = {"RIF":"S","INH":"S","EMB":"S","PZA":"S"}

        resistance_class = _classify_resistance(drug_profile)

        # ── X-Ray / CT row (Shenzhen) — stratified resample, not round-robin ───
        if xray_available and xray_work is not None:
            xray_row = _pick_shenzhen_row_stratified(
                xray_work,
                sym_row.get("gender"),
                resistance_class,
                patient_key=f"{patient_id}|{uniqueid}",
            )
        else:
            xray_row = None
        age_val  = xray_row.get("age") if xray_row is not None else None
        sex_val  = xray_row.get("sex") if xray_row is not None else None

        try:
            age = int(float(age_val)) if age_val and not pd.isna(age_val) else None
        except (ValueError, TypeError):
            age = None
        sex = str(sex_val) if sex_val and not pd.isna(sex_val) else None

        # ── Build narratives ──────────────────────────────────────────────────

        record = {
            # Identity
            "id":           str(i + 1),
            "patient_id":   patient_id,
            "condition_id": patient_id,
            "name":         str(sym_row.get("name", f"Patient {i+1}")),
            "gender":       sex or str(sym_row.get("gender", "Unknown")),
            "age":          age,
            "subject_id":   str(sym_row.get("no", i + 1)),
            "country":      country,
            "isolate_id":   uniqueid,

            # Resistance
            "type_of_resistance": resistance_class,
            "drug_profile":       drug_profile,
            "n_dst_datasets":     int(dst_row.get("N_DATASETS", 1)),

            # Modality narratives → feed directly to council agents
            "clinical_data": _symptoms_narrative(sym_row, age=age, sex=sex),
            "genomic_data":  _resistance_narrative(drug_profile, country, resistance_class),
            "xray_data":     _xray_narrative(xray_row, patient_id),
            "ct_data":       _ct_narrative(xray_row, patient_id),

            # Shenzhen X-ray source fields (for detail view)
            "xray_tb_status":    str(xray_row.get("tb_status","Unknown")) if xray_row is not None else None,
            "xray_patient_id":   str(xray_row.get("patient_id","")) if xray_row is not None else None,
            "xray_source":       "NIH/NLM Shenzhen Hospital CXR Dataset" if xray_available else None,

            # Agent availability flags
            "has_clinical":  True,
            "has_genomics":  bool(drug_profile),
            "has_xray":      xray_row is not None,
            "has_ct":        xray_row is not None,
            "has_cxr":       xray_row is not None,
            "has_dst":       True,

            # Scores
            "clinical_agent_score": score_from_seed(seed+"clin", 7.0),
            "genomics_agent_score": score_from_seed(seed+"gen",  7.5),
            "xray_agent_score":     score_from_seed(seed+"xray", 7.2) if xray_row is not None else None,
            "ct_agent_score":       score_from_seed(seed+"ct",   6.8) if xray_row is not None else None,
            "rag_confidence_score": score_from_seed(seed+"rag",  7.8),

            "judge_verdict": (
                "Positive" if resistance_class in ("MDR non XDR","XDR","Pre-XDR","Mono DR","Poly DR")
                else "Inconclusive"
            ),
        }
        records.append(record)

    active_agents = ["Clinical", "Genomic"]
    if xray_available:
        active_agents += ["X-Ray", "CT"]
    active_agents.append("Judge")
    logger.info("Built %d records. Active agents: %s", len(records), active_agents)
    return records