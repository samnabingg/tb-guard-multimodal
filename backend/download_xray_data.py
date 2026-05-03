"""
download_xray_data.py  —  Run once to fetch free TB imaging data
===================================================================
Run from backend/:   python download_xray_data.py

Downloads from NIH National Library of Medicine (no login required):
  - Shenzhen Hospital TB CXR Dataset
    Source: National Library of Medicine / NIH
    URL: https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/
    336 TB patients, radiologist-annotated findings (19 abnormality types)
    CSV + per-patient clinical readings (age, sex, TB status, lung findings text)

Output files written to data/:
  shenzhen_xray_stats.csv     — per-patient abnormality summary (CXR findings)
  shenzhen_clinical.csv       — per-patient age, sex, TB+/- label, clinical notes
"""

import os
import urllib.request
import json
import re
import time
import csv
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ── NIH NLM public URLs (no auth required) ────────────────────────────────────
BASE = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set"
STATS_CSV_URL     = f"{BASE}/Annotations/Statistics_ShenzhenDataset.csv"
CLINICAL_DIR_URL  = f"{BASE}/ClinicalReadings/index.html"
ANNOTATIONS_INDEX = f"{BASE}/Annotations/Annotations_json/index.html"

HEADERS = {"User-Agent": "TB-Guard-Research/1.0 (educational TB screening project)"}


def _fetch(url: str, label: str) -> bytes:
    print(f"  Downloading {label}...")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    print(f"  ✓ {len(data):,} bytes")
    return data


def download_stats_csv():
    """Download the per-patient abnormality statistics CSV."""
    out = DATA_DIR / "shenzhen_xray_stats.csv"
    if out.exists():
        print(f"  Skipping (already exists): {out}")
        return
    data = _fetch(STATS_CSV_URL, "Shenzhen abnormality stats CSV")
    out.write_bytes(data)
    print(f"  Saved → {out}")


def download_clinical_readings():
    """
    Download per-patient clinical reading text files from the ClinicalReadings folder.
    Each file is a small .txt with: patient ID, age, sex, TB status, radiologist notes.
    Aggregates all into shenzhen_clinical.csv.
    """
    out = DATA_DIR / "shenzhen_clinical.csv"
    if out.exists():
        print(f"  Skipping (already exists): {out}")
        return

    # Fetch directory index to get list of .txt files
    print("  Fetching ClinicalReadings index...")
    req = urllib.request.Request(CLINICAL_DIR_URL, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            index_html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  Warning: Could not fetch clinical index: {e}")
        print("  Generating synthetic clinical readings from stats CSV instead...")
        _generate_clinical_from_stats()
        return

    # Parse file list from directory index
    txt_files = re.findall(r'href="([^"]+\.txt)"', index_html)
    if not txt_files:
        print("  Warning: No .txt files found in index. Generating from stats CSV...")
        _generate_clinical_from_stats()
        return

    print(f"  Found {len(txt_files)} clinical reading files")
    records = []

    for i, fname in enumerate(txt_files[:350]):   # cap at 350
        file_url = f"{BASE}/ClinicalReadings/{fname}"
        try:
            req = urllib.request.Request(file_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                text = resp.read().decode("utf-8", errors="replace").strip()
            records.append(_parse_clinical_txt(fname, text))
            if (i + 1) % 50 == 0:
                print(f"    Downloaded {i+1}/{len(txt_files[:350])}...")
            time.sleep(0.05)   # be polite to NIH servers
        except Exception as e:
            print(f"    Warning: Could not fetch {fname}: {e}")

    if records:
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"  ✓ Saved {len(records)} clinical records → {out}")
    else:
        print("  No records fetched. Generating from stats CSV...")
        _generate_clinical_from_stats()


def _parse_clinical_txt(fname: str, text: str) -> dict:
    """
    Parse a Shenzhen clinical reading text file.
    Format varies but typically includes age, sex, TB+/- and clinical notes.
    """
    patient_id = fname.replace(".txt", "")
    age, sex, tb_status, notes = None, None, None, text

    # Try to extract structured fields from common formats
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r"^(age|Age)[:\s]+(\d+)", line):
            m = re.search(r"(\d+)", line)
            if m:
                age = int(m.group(1))
        elif re.match(r"^(sex|Sex|gender|Gender)[:\s]+", line, re.I):
            sex = "Female" if re.search(r"f(emale)?", line, re.I) else "Male"
        elif re.search(r"TB[+\-]|tuberculosis", line, re.I):
            tb_status = "Positive" if re.search(r"TB\+|positive", line, re.I) else "Negative"

    # Patient ID encodes TB status: CHNCXR_XXXX_0 = normal, CHNCXR_XXXX_1 = TB
    if tb_status is None and "_1." in fname:
        tb_status = "Positive"
    elif tb_status is None and "_0." in fname:
        tb_status = "Negative"

    return {
        "patient_id":  patient_id,
        "age":         age,
        "sex":         sex,
        "tb_status":   tb_status or "Unknown",
        "clinical_text": text[:1000],
    }


def _generate_clinical_from_stats():
    """
    If ClinicalReadings aren't reachable, build a clinical CSV from
    the stats CSV (which has patient IDs and abnormality counts).
    """
    stats_path = DATA_DIR / "shenzhen_xray_stats.csv"
    out = DATA_DIR / "shenzhen_clinical.csv"

    if not stats_path.exists():
        print("  Stats CSV not found either. Skipping clinical generation.")
        return

    import csv as csv_mod
    records = []
    with open(stats_path, newline="", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            pid = row.get("Patient", row.get("patient_id", row.get("ID", "")))
            # Infer TB status from filename convention (_1 = TB positive)
            tb = "Positive" if str(pid).endswith("_1") else "Negative"
            records.append({
                "patient_id":    pid,
                "age":           None,
                "sex":           None,
                "tb_status":     tb,
                "clinical_text": f"Shenzhen CXR patient {pid}. TB status: {tb}.",
            })

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["patient_id","age","sex","tb_status","clinical_text"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  ✓ Generated {len(records)} records → {out}")


def verify_downloads():
    """Print a summary of what was downloaded."""
    print("\n── Download summary ──────────────────────────────────────────")
    for fname in ["shenzhen_xray_stats.csv", "shenzhen_clinical.csv"]:
        p = DATA_DIR / fname
        if p.exists():
            import csv as c
            with open(p) as f:
                rows = sum(1 for _ in c.reader(f)) - 1
            print(f"  ✓ {fname}: {rows} rows, {p.stat().st_size:,} bytes")
        else:
            print(f"  ✗ {fname}: NOT FOUND")

    print("\nNext: restart your backend. data_loader.py will auto-detect these files.")


if __name__ == "__main__":
    print("TB-Guard: Downloading NIH Shenzhen TB CXR data (free, no login required)")
    print("=" * 65)

    print("\n[1/2] Abnormality statistics CSV")
    download_stats_csv()

    print("\n[2/2] Per-patient clinical readings")
    download_clinical_readings()

    verify_downloads()