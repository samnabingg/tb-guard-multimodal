# TB-Guard: Multi-Agent AI System for Explainable Tuberculosis Detection

TB-Guard is a multi-agent artificial intelligence framework designed to improve the reliability and explainability of tuberculosis (TB) detection from chest X-rays. The system simulates a structured diagnostic debate among multiple large language models (LLMs) and produces a final consensus decision with justification.

---

## Overview

Conventional AI systems typically generate a single prediction with limited transparency. TB-Guard addresses this limitation by introducing a council-based architecture in which multiple models independently analyze a case, challenge each other’s reasoning, and produce a final, explainable decision.

---

## System Architecture

The system is composed of three primary agent roles:

### Prognosis Agents

These agents perform the initial TB likelihood assessment and provide independent reasoning based on the input.

### Devil’s Advocate Agents

These agents critically evaluate the initial diagnoses, propose alternative interpretations, and identify missing or insufficient evidence.

### Judge / Synthesizer Agent

This agent reviews the full debate, reconciles conflicting arguments, and produces a final diagnosis along with an explanation and confidence score.

---

## Model Assignments

| Role                         | Model            | Provider         |
| ---------------------------- | ---------------- | ---------------- |
| Prognosis (Primary)          | GPT-4o           | GitHub Models    |
| Prognosis (Secondary)        | Gemini 2.5 Flash | Google AI Studio |
| Devil’s Advocate (Primary)   | Llama 3.3 70B    | Groq             |
| Devil’s Advocate (Secondary) | Qwen3 32B        | Groq             |
| Judge / Synthesizer          | GPT-OSS 120B     | Groq             |

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

## Methodology

TB-Guard is based on a multi-agent debate paradigm combined with API-driven large language model inference. The system workflow is as follows:

1. Prognosis agents independently generate initial diagnostic assessments.
2. Devil’s Advocate agents critique these assessments and introduce alternative reasoning.
3. The Judge agent synthesizes the discussion into a final decision.

This architecture improves robustness by encouraging disagreement and structured reasoning before producing an output.

---

## Key Features

* Multi-agent consensus-based decision making
* Explainable outputs through structured debate
* Integration of multiple model providers (OpenAI, Google, Groq)
* Modular and extensible backend design
* Real-time API-based evaluation

---

## Limitations

* Not intended for clinical or medical use
* Dependent on the reasoning quality of language models
* No direct image processing integration at this stage
* Susceptible to hallucinations or inconsistent reasoning

---

## Future Work

* Integration of convolutional neural networks for X-ray analysis
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
# TB-Guard: Multi-Agent AI System for Explainable Tuberculosis Detection

TB-Guard is a multi-agent artificial intelligence framework designed to improve the reliability and explainability of tuberculosis (TB) detection from chest X-rays. The system simulates a structured diagnostic debate among multiple large language models (LLMs) and produces a final consensus decision with justification.

---

## Overview

Conventional AI systems typically generate a single prediction with limited transparency. TB-Guard addresses this limitation by introducing a council-based architecture in which multiple models independently analyze a case, challenge each other’s reasoning, and produce a final, explainable decision.

---

## System Architecture

The system is composed of three primary agent roles:

### Prognosis Agents

These agents perform the initial TB likelihood assessment and provide independent reasoning based on the input.

### Devil’s Advocate Agents

These agents critically evaluate the initial diagnoses, propose alternative interpretations, and identify missing or insufficient evidence.

### Judge / Synthesizer Agent

This agent reviews the full debate, reconciles conflicting arguments, and produces a final diagnosis along with an explanation and confidence score.

---

## Model Assignments

| Role                         | Model            | Provider         |
| ---------------------------- | ---------------- | ---------------- |
| Prognosis (Primary)          | GPT-4o           | GitHub Models    |
| Prognosis (Secondary)        | Gemini 2.5 Flash | Google AI Studio |
| Devil’s Advocate (Primary)   | Llama 3.3 70B    | Groq             |
| Devil’s Advocate (Secondary) | Qwen3 32B        | Groq             |
| Judge / Synthesizer          | GPT-OSS 120B     | Groq             |

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

## Methodology

TB-Guard is based on a multi-agent debate paradigm combined with API-driven large language model inference. The system workflow is as follows:

1. Prognosis agents independently generate initial diagnostic assessments.
2. Devil’s Advocate agents critique these assessments and introduce alternative reasoning.
3. The Judge agent synthesizes the discussion into a final decision.

This architecture improves robustness by encouraging disagreement and structured reasoning before producing an output.

---

## Key Features

* Multi-agent consensus-based decision making
* Explainable outputs through structured debate
* Integration of multiple model providers (OpenAI, Google, Groq)
* Modular and extensible backend design
* Real-time API-based evaluation

---

## Limitations

* Not intended for clinical or medical use
* Dependent on the reasoning quality of language models
* No direct image processing integration at this stage
* Susceptible to hallucinations or inconsistent reasoning

---

## Future Work

* Integration of convolutional neural networks for X-ray analysis
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
