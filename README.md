# TB-Guard: Multi-Agent AI System for Explainable Tuberculosis Detection

TB-Guard is a multi-agent artificial intelligence framework designed to improve the reliability and explainability of tuberculosis (TB) assessment using multimodal patient data. The system combines specialized AI agents and produces a final integrated decision with justification.

---

## Overview

Conventional AI systems typically generate a single prediction with limited transparency. TB-Guard addresses this limitation by introducing a council-based architecture in which multiple models independently analyze modality-specific data and produce a final, explainable decision.

---

## System Architecture

The system is composed of five specialist agents:

### Clinical Agent

Analyzes clinical records, symptoms, history, and metadata for TB risk indicators.

### X-Ray Agent

Interprets chest X-ray findings (text-described radiology features) for TB-consistent imaging patterns.

### Genomic Agent

Interprets genomic mutation and resistance marker information for TB drug-resistance profiling.

### CT Agent

Analyzes chest CT findings for high-resolution structural patterns consistent with TB.

### Integration Agent

Synthesizes outputs from the four modality agents, resolves conflicts, and issues the final TB verdict with confidence.

---

## Model Assignments

| Agent             | Model            | Provider      |
| ----------------- | ---------------- | ------------- |
| Clinical Agent    | GPT-4o           | GitHub Models |
| Genomic Agent     | Gemini 2.5 Flash | Google AI     |
| CT Agent          | Llama 3.3 70B    | Groq          |
| X-Ray Agent       | Qwen3 32B        | Groq          |
| Integration Agent | GPT-OSS 120B     | Groq          |

---

## Project Structure

```
tb-guard/
│
├── backend/
│   ├── main.py
│   └── api/
│       └── test_connections.py
│
├── .env                # Local environment variables (not tracked)
├── .env.example        # Environment template
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/samnabingg/TB-Guard-A-Multi-Agent-AI-System-for-Explainable-Tuberculosis-Detection-via-Chest-X-Rays.git
cd tb-guard
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```
GITHUB_TOKEN=your_token_here
GOOGLE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

Important: Do not commit the `.env` file to version control.

---

## Testing

To verify that all agents are correctly configured and reachable:

```bash
cd backend/api
python test_connections.py
```

A successful run should confirm connectivity for all five agents.

---

## API Usage

Run the backend API from the `backend` folder:

```bash
uvicorn main:app --reload
```

Base URL:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### Analyze direct case input

Endpoint:

```text
POST /analyze-case
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/analyze-case \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "case-001",
    "clinical_data": "3-month cough, night sweats, weight loss, AFB positive.",
    "genomic_data": "rpoB S450L present; inhA promoter mutation present.",
    "ct_data": "No CT available.",
    "xray_data": "Right upper lobe cavitary consolidation, consistent with post-primary TB."
  }'
```

### Analyze TB-DEPOT-style input

Endpoint:

```text
POST /analyze-tbdepot-case
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/analyze-tbdepot-case \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id": "tbdepot-123",
    "clinical_notes": "Persistent cough, fever, weight loss.",
    "symptoms": "Night sweats, hemoptysis",
    "treatment_history": "Prior TB treatment 2 years ago",
    "wgs_data": "rpoB mutation detected",
    "ct_findings": "Upper lobe nodular infiltrates and cavitation",
    "cxr_report": "Bilateral upper lobe opacities"
  }'
```

---

## Methodology

TB-Guard is based on a multi-agent multimodal reasoning workflow combined with API-driven large language model inference. The system workflow is as follows:

1. The Clinical, X-Ray, Genomic, and CT agents independently analyze available modality data.
2. Each agent returns a structured conclusion and confidence signal.
3. The Integration agent synthesizes all conclusions into a final TB decision.
4. If evidence is insufficient or conflicting, the system can request additional RAG context and retry synthesis.

This architecture improves robustness by combining complementary evidence sources before producing an output.

---

## Key Features

* Five-agent multimodal decision architecture
* Explainable outputs through structured agent synthesis
* Integration of multiple model providers (OpenAI, Google, Groq)
* Designed for TB-DEPOT-style multimodal case inputs
* Modular and extensible backend design
* Real-time API-based evaluation

---

## Limitations

* Not intended for clinical or medical use
* Dependent on the reasoning quality of language models
* Current imaging agents consume text-derived radiology descriptions (not raw DICOM/image pixels)
* Susceptible to hallucinations or inconsistent reasoning

---

## Future Work

* Direct integration with TB-DEPOT ingestion and schema mapping
* Integration of vision models for direct image understanding
* Retrieval-augmented generation using medical literature
* Development of a full-stack interface (React + FastAPI)
* Deployment as a research prototype for decision support systems

---

## References

* World Health Organization. Global Tuberculosis Report (2023)
* Rajpurkar et al. CheXNet (2017)
* Park et al. Generative Agents (2023)
* Liang et al. LLM Debate (2024)

---

## Authors

* Samana Dahal
* Project Team Members

---

## License

This project is developed for academic and research purposes.
