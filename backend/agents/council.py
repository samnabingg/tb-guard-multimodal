import os, sys
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.append(os.path.join(os.path.dirname(__file__), "../rag"))
from rag_pipeline import retrieve

load_dotenv()

github_client = OpenAI(
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com"
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gemini_llm  = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# ─────────────────────────────────────────────
# CLINICAL DATA AGENT — GPT-4o
# Fires only if clinical data is present
# ─────────────────────────────────────────────
def _retrieve_context(query: str, vector_store, k: int = 4) -> str:
    results = retrieve(query, vector_store, k=k)
    return "\n\n".join(r.page_content for r in results)


def clinical_data_agent(clinical_data: str, vector_store) -> dict:
    rag_context = _retrieve_context(
        "TB clinical presentation symptoms risk factors treatment history",
        vector_store
    )
    prompt = f"""You are a specialist in TB clinical data interpretation.

CLINICAL DATA:
{clinical_data}

RELEVANT MEDICAL LITERATURE:
{rag_context}

Analyze this clinical data for TB. Provide:
1. TB likelihood: low / moderate / high
2. Key clinical indicators supporting your conclusion
3. MDR-TB or XDR-TB risk based on treatment history and demographics
4. Confidence score (0-100)
5. What additional clinical data would strengthen this assessment

Be specific. Cite the literature provided where relevant."""

    response = github_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return {
        "agent": "Clinical Data Agent (GPT-4o)",
        "conclusion": response.choices[0].message.content
    }


# ─────────────────────────────────────────────
# DNA / GENOMIC AGENT — Gemini 2.5 Flash
# Fires only if genomic data is present
# ─────────────────────────────────────────────
def dna_genomic_agent(genomic_data: str, vector_store) -> dict:
    rag_context = _retrieve_context(
        "MDR XDR tuberculosis resistance markers genomic mutations rifampicin isoniazid",
        vector_store
    )
    prompt = f"""You are a specialist in TB genomic and resistance analysis.

GENOMIC / LAB DATA:
{genomic_data}

RELEVANT LITERATURE ON RESISTANCE MARKERS:
{rag_context}

Analyze this genomic data for TB drug resistance. Provide:
1. TB likelihood based on genomic evidence
2. Drug resistance profile: susceptible / MDR-TB / XDR-TB / pre-XDR
3. Specific resistance markers identified (rpoB, katG, inhA, etc.)
4. Confidence score (0-100)
5. Which resistance markers are missing that would complete the picture

Reference the literature where relevant."""

    response = gemini_llm.invoke(prompt)
    return {
        "agent": "DNA/Genomic Agent (Gemini 2.5 Flash)",
        "conclusion": response.content
    }


# ─────────────────────────────────────────────
# CT AGENT — Llama 3.3 70B
# Fires only if CT scan data is present
# ─────────────────────────────────────────────
def ct_agent(ct_data: str, vector_store) -> dict:
    rag_context = _retrieve_context(
        "tuberculosis CT scan findings cavitation nodules consolidation",
        vector_store
    )
    prompt = f"""You are a radiologist specializing in CT analysis for TB detection.

CT SCAN FINDINGS:
{ct_data}

RELEVANT RADIOLOGY LITERATURE:
{rag_context}

Analyze these CT findings for TB. Provide:
1. TB likelihood: low / moderate / high
2. Key CT features supporting your conclusion (cavitation, tree-in-bud, consolidation, etc.)
3. Distribution pattern (upper lobe predominance typical of TB?)
4. Alternative diagnoses to consider from these CT findings
5. Confidence score (0-100)"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return {
        "agent": "CT Agent (Llama 3.3 70B)",
        "conclusion": response.choices[0].message.content
    }


# ─────────────────────────────────────────────
# X-RAY AGENT — Qwen3 32B
# Fires only if X-ray data/description is present
# Note: Using text description of X-ray findings.
# Vision model integration is a future enhancement.
# ─────────────────────────────────────────────
def xray_agent(xray_data: str, vector_store) -> dict:
    rag_context = _retrieve_context(
        "tuberculosis chest X-ray radiological signs infiltrates cavitation upper lobe",
        vector_store
    )
    prompt = f"""You are a radiologist specializing in chest X-ray interpretation for TB.

CHEST X-RAY FINDINGS:
{xray_data}

RELEVANT RADIOLOGY LITERATURE:
{rag_context}

Analyze these X-ray findings for TB. Provide:
1. TB likelihood: low / moderate / high
2. Key radiological features (location, infiltrates, cavities, pleural effusion, etc.)
3. CXR pattern consistency with primary vs. reactivation TB
4. Differential diagnoses from X-ray alone
5. Confidence score (0-100)"""

    response = groq_client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return {
        "agent": "X-Ray Agent (Qwen3 32B)",
        "conclusion": response.choices[0].message.content
    }


# ─────────────────────────────────────────────
# INTEGRATION / JUDGE AGENT — GPT-OSS 120B
# Receives all available agent conclusions,
# checks for consensus, loops back if uncertain
# ─────────────────────────────────────────────
def judge_agent(agent_conclusions: list[dict], vector_store, rag_context: str = "") -> dict:
    # Build the debate transcript from whichever agents ran
    transcript = "\n\n".join(
        f"--- {c['agent']} ---\n{c['conclusion']}"
        for c in agent_conclusions
    )

    if not rag_context:
        rag_context = _retrieve_context(
            "WHO tuberculosis diagnosis guidelines treatment criteria consensus",
            vector_store
        )

    prompt = f"""You are the Integration and Judge Agent for the TB-Guard Council of AI.
You have received conclusions from {len(agent_conclusions)} specialist agent(s).

AGENT CONCLUSIONS:
{transcript}

SUPPORTING EVIDENCE (WHO GUIDELINES + LITERATURE):
{rag_context}

Your task:

STEP 1 — CONSENSUS CHECK:
Do the available agents broadly agree on TB likelihood and drug-resistance classification?
Answer: CONSENSUS / NO CONSENSUS / INSUFFICIENT DATA

STEP 2 — If CONSENSUS or sufficient data, issue your final verdict:

FINAL VERDICT: [TB Positive / TB Negative / Inconclusive]
TB TYPE: [Standard TB / MDR-TB / XDR-TB / Insufficient data]
CONFIDENCE: [0-100]

REASONING:
[How you weighed each agent's input. Where they agreed and where they conflicted.]

POINTS OF AGREEMENT:
[What the agents agreed on]

POINTS OF CONFLICT:
[Where agents disagreed and how you resolved it]

RECOMMENDED NEXT STEPS:
[Clinical actions following from this verdict]

CITED SOURCES:
[List source tags from the literature above]

STEP 3 — If NO CONSENSUS:
State exactly what is missing or conflicting, and what specific RAG query
would resolve it. Format: NEEDS_RAG: <your query string>"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    verdict_text = response.choices[0].message.content

    # Detect if judge is requesting another RAG pass
    needs_rag = "NEEDS_RAG:" in verdict_text
    rag_query = ""
    if needs_rag:
        rag_query = verdict_text.split("NEEDS_RAG:")[-1].strip().split("\n")[0]

    return {
        "verdict_text": verdict_text,
        "needs_rag": needs_rag,
        "rag_query": rag_query,
        "agent_conclusions": agent_conclusions
    }