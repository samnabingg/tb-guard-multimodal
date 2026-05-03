"""
council.py  —  TB-Guard Council of AI
LangChain-integrated multi-agent orchestration with RunnableParallel.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  RunnableParallel  (all 4 agents run simultaneously) │
  │  ┌──────────┐ ┌─────────┐ ┌──────┐ ┌──────────┐    │
  │  │ Clinical │ │ Genomic │ │  CT  │ │  X-Ray   │    │
  │  │  GPT-4o  │ │ Gemini  │ │Llama │ │  Qwen3   │    │
  │  └──────────┘ └─────────┘ └──────┘ └──────────┘    │
  └───────────────────┬─────────────────────────────────┘
                      │  (conclusions dict)
                      ▼
            ┌─────────────────┐
            │  Judge Agent    │
            │  GPT-OSS 120B   │
            │  (consensus     │
            │   + RAG retry)  │
            └─────────────────┘

Fixes applied (v2):
  1. _safe_invoke() — wraps every agent so a single auth/network
     failure returns a degraded string instead of crashing
     RunnableParallel and returning HTTP 500.
  2. Lambda closure bug fixed — each lambda now receives its own
     captured copy of `rag_ctx` and `chain` via default arguments,
     preventing all lambdas from accidentally sharing the last
     assigned value of the loop variable.
"""

import os
import sys
from dotenv import load_dotenv

# ── LangChain imports ──────────────────────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# ── RAG pipeline ───────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "../rag"))
from rag_pipeline import retrieve, load_vector_store

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# LLM CLIENTS  (LangChain wrappers — one per agent)
# ══════════════════════════════════════════════════════════════════════════════

# Clinical Agent — GPT-4o via GitHub Models (OpenAI-compatible endpoint)
clinical_llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com",
    temperature=0.3,
)

# Genomic Agent — Gemini 2.5 Flash
genomic_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)

# CT Agent — Llama 3.3 70B via Groq
ct_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)

# X-Ray Agent — Qwen3 32B via Groq
xray_llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)

# Judge / Integration Agent — GPT-OSS 120B via Groq
judge_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _retrieve_context(query: str, vector_store, k: int = 4) -> str:
    """Retrieve RAG context chunks and join them into a single string."""
    results = retrieve(query, vector_store, k=k)
    return "\n\n".join(r.page_content for r in results)


def _safe_invoke(agent_name: str, chain, inputs: dict) -> str:
    """
    FIX 1: Fault-tolerant agent wrapper.

    Wraps any LangChain chain invocation so that a single agent
    failure (401 auth, timeout, API quota, etc.) returns a degraded
    placeholder string instead of raising an exception that would
    crash the entire RunnableParallel and return HTTP 500.

    The Judge Agent is designed to handle UNAVAILABLE agents
    gracefully by marking them as missing data rather than failing.
    """
    try:
        result = chain.invoke(inputs)
        # chain already ends with StrOutputParser so result is a plain str,
        # but guard against AIMessage in case chain is swapped later.
        return result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        error_type = type(e).__name__
        short_msg = str(e)[:200]
        print(f"[WARN] {agent_name} agent FAILED ({error_type}): {short_msg}")
        # Return a structured degraded response the Judge can read
        return (
            f"[{agent_name.upper()} AGENT UNAVAILABLE]\n"
            f"Reason: {error_type} — {short_msg}\n"
            f"This agent's modality is excluded from the consensus. "
            f"Judge should proceed with remaining agents and flag this gap."
        )


def _has_data(val) -> bool:
    """
    Returns True only if the modality field contains real data.
    Rejects None, empty string, whitespace-only, and common
    placeholder strings that data_loader emits when data is absent.
    """
    if not val:
        return False
    s = str(val).strip().lower()
    if not s:
        return False
    # Reject known placeholder patterns from data_loader
    EMPTY_SIGNALS = {"none", "n/a", "not available", "no data", "missing", "unavailable", "nan"}
    return s not in EMPTY_SIGNALS

def _build_agent_chain(llm, prompt_template: ChatPromptTemplate, agent_name: str):
    """
    Returns a LangChain LCEL chain:
        prompt | llm | StrOutputParser
    wrapped so it returns {"agent": agent_name, "conclusion": <text>}
    """
    parser = StrOutputParser()
    chain = prompt_template | llm | parser

    def run_and_tag(inputs: dict) -> dict:
        conclusion = chain.invoke(inputs)
        return {"agent": agent_name, "conclusion": conclusion}

    return RunnableLambda(run_and_tag)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

clinical_prompt = ChatPromptTemplate.from_template("""You are a specialist in TB clinical data interpretation.

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

Be specific. Cite the literature provided where relevant.""")

genomic_prompt = ChatPromptTemplate.from_template("""You are a specialist in TB genomic and resistance analysis.

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

Reference the literature where relevant.""")

ct_prompt = ChatPromptTemplate.from_template("""You are a radiologist specializing in CT analysis for TB detection.

CT SCAN FINDINGS:
{ct_data}

RELEVANT RADIOLOGY LITERATURE:
{rag_context}

Analyze these CT findings for TB. Provide:
1. TB likelihood: low / moderate / high
2. Key CT features supporting your conclusion (cavitation, tree-in-bud, consolidation, etc.)
3. Distribution pattern (upper lobe predominance typical of TB?)
4. Alternative diagnoses to consider from these CT findings
5. Confidence score (0-100)""")

xray_prompt = ChatPromptTemplate.from_template("""You are a radiologist specializing in chest X-ray interpretation for TB.

CHEST X-RAY FINDINGS:
{xray_data}

RELEVANT RADIOLOGY LITERATURE:
{rag_context}

Analyze these X-ray findings for TB. Provide:
1. TB likelihood: low / moderate / high
2. Key radiological features (location, infiltrates, cavities, pleural effusion, etc.)
3. CXR pattern consistency with primary vs. reactivation TB
4. Differential diagnoses from X-ray alone
5. Confidence score (0-100)""")

judge_prompt = ChatPromptTemplate.from_template("""You are the Integration and Judge Agent for the TB-Guard Council of AI.
You have received conclusions from {agent_count} specialist agent(s).

NOTE: Some agents may show [AGENT UNAVAILABLE] — treat these as missing modalities,
not as negative findings. Base your verdict only on the agents that responded.

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

UNAVAILABLE AGENTS:
[List any agents that were unavailable and what gap this leaves in the assessment]

RECOMMENDED NEXT STEPS:
[Clinical actions following from this verdict]

CITED SOURCES:
[List source tags from the literature above]

STEP 3 — If NO CONSENSUS:
State exactly what is missing or conflicting, and what specific RAG query
would resolve it. Format: NEEDS_RAG: <your query string>""")


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL AGENT FUNCTIONS  (LangChain chains under the hood)
# ══════════════════════════════════════════════════════════════════════════════

def clinical_data_agent(clinical_data: str, vector_store) -> dict:
    """Clinical Agent — GPT-4o via GitHub Models."""
    rag_context = _retrieve_context(
        "TB clinical presentation symptoms risk factors treatment history",
        vector_store
    )
    chain = _build_agent_chain(
        clinical_llm, clinical_prompt, "Clinical Data Agent (GPT-4o)"
    )
    return chain.invoke({"clinical_data": clinical_data, "rag_context": rag_context})


def dna_genomic_agent(genomic_data: str, vector_store) -> dict:
    """Genomic Agent — Gemini 2.5 Flash."""
    rag_context = _retrieve_context(
        "MDR XDR tuberculosis resistance markers genomic mutations rifampicin isoniazid",
        vector_store
    )
    chain = _build_agent_chain(
        genomic_llm, genomic_prompt, "DNA/Genomic Agent (Gemini 2.5 Flash)"
    )
    return chain.invoke({"genomic_data": genomic_data, "rag_context": rag_context})


def ct_agent(ct_data: str, vector_store) -> dict:
    """CT Agent — Llama 3.3 70B via Groq."""
    rag_context = _retrieve_context(
        "tuberculosis CT scan findings cavitation nodules consolidation",
        vector_store
    )
    chain = _build_agent_chain(ct_llm, ct_prompt, "CT Agent (Llama 3.3 70B)")
    return chain.invoke({"ct_data": ct_data, "rag_context": rag_context})


def xray_agent(xray_data: str, vector_store) -> dict:
    """X-Ray Agent — Qwen3 32B via Groq."""
    rag_context = _retrieve_context(
        "tuberculosis chest X-ray radiological signs infiltrates cavitation upper lobe",
        vector_store
    )
    chain = _build_agent_chain(xray_llm, xray_prompt, "X-Ray Agent (Qwen3 32B)")
    return chain.invoke({"xray_data": xray_data, "rag_context": rag_context})


# ══════════════════════════════════════════════════════════════════════════════
# JUDGE AGENT  (LangChain chain)
# ══════════════════════════════════════════════════════════════════════════════

def judge_agent(agent_conclusions: list[dict], vector_store, rag_context: str = "") -> dict:
    """
    Integration / Judge Agent — GPT-OSS 120B via Groq.
    Receives all specialist conclusions, checks consensus, loops if needed.
    """
    transcript = "\n\n".join(
        f"--- {c['agent']} ---\n{c['conclusion']}"
        for c in agent_conclusions
    )

    if not rag_context:
        rag_context = _retrieve_context(
            "WHO tuberculosis diagnosis guidelines treatment criteria consensus",
            vector_store
        )

    judge_chain = judge_prompt | judge_llm | StrOutputParser()

    verdict_text = judge_chain.invoke({
        "agent_count": len(agent_conclusions),
        "transcript": transcript,
        "rag_context": rag_context,
    })

    needs_rag = "NEEDS_RAG:" in verdict_text
    rag_query = ""
    if needs_rag:
        rag_query = verdict_text.split("NEEDS_RAG:")[-1].strip().split("\n")[0]

    return {
        "verdict_text": verdict_text,
        "needs_rag": needs_rag,
        "rag_query": rag_query,
        "agent_conclusions": agent_conclusions,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL COUNCIL RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_council_parallel(patient_case: dict) -> dict:
    """
    Builds a RunnableParallel from whichever modalities are present,
    runs all available specialist agents simultaneously, then passes
    their combined conclusions to the Judge Agent.

    Two bugs fixed vs v1:
      - FIX 1: _safe_invoke wraps every agent branch so one failure
        does not crash the whole parallel runner.
      - FIX 2: Lambda default-argument capture (`chain=chain, ctx=rag_ctx`)
        ensures each lambda holds its OWN snapshot of chain and rag_ctx
        instead of all sharing the last value assigned to the variable
        (classic Python late-binding closure trap).
    """
    print("\n[TB-GUARD] Council of AI — Starting parallel deliberation...\n")
    vector_store = load_vector_store()

    parallel_branches = {}

    # ── Clinical ───────────────────────────────────────────────────────────────
    if _has_data(patient_case.get("clinical_data")):
        rag_ctx = _retrieve_context(
            "TB clinical presentation symptoms risk factors treatment history",
            vector_store
        )
        clinical_chain = clinical_prompt | clinical_llm | StrOutputParser()

        # FIX 2: capture chain and ctx NOW via default args, not at call time
        parallel_branches["clinical"] = RunnableLambda(
            lambda _, chain=clinical_chain, ctx=rag_ctx: _safe_invoke(
                "clinical", chain,
                {
                    "clinical_data": patient_case["clinical_data"],
                    "rag_context": ctx,
                }
            )
        )

    # ── Genomic ────────────────────────────────────────────────────────────────
    if _has_data(patient_case.get("genomic_data")):
        rag_ctx = _retrieve_context(
            "MDR XDR tuberculosis resistance markers genomic mutations rifampicin isoniazid",
            vector_store
        )
        genomic_chain = genomic_prompt | genomic_llm | StrOutputParser()

        parallel_branches["genomic"] = RunnableLambda(
            lambda _, chain=genomic_chain, ctx=rag_ctx: _safe_invoke(
                "genomic", chain,
                {
                    "genomic_data": patient_case["genomic_data"],
                    "rag_context": ctx,
                }
            )
        )

    # ── CT ─────────────────────────────────────────────────────────────────────
    if _has_data(patient_case.get("ct_data")):
        rag_ctx = _retrieve_context(
            "tuberculosis CT scan findings cavitation nodules consolidation",
            vector_store
        )
        ct_chain = ct_prompt | ct_llm | StrOutputParser()

        parallel_branches["ct"] = RunnableLambda(
            lambda _, chain=ct_chain, ctx=rag_ctx: _safe_invoke(
                "ct", chain,
                {
                    "ct_data": patient_case["ct_data"],
                    "rag_context": ctx,
                }
            )
        )

    # ── X-Ray ──────────────────────────────────────────────────────────────────
    if _has_data(patient_case.get("xray_data")):
        rag_ctx = _retrieve_context(
            "tuberculosis chest X-ray radiological signs infiltrates cavitation upper lobe",
            vector_store
        )
        xray_chain = xray_prompt | xray_llm | StrOutputParser()

        parallel_branches["xray"] = RunnableLambda(
            lambda _, chain=xray_chain, ctx=rag_ctx: _safe_invoke(
                "xray", chain,
                {
                    "xray_data": patient_case["xray_data"],
                    "rag_context": ctx,
                }
            )
        )

    if not parallel_branches:
        print("[ERROR] No modality data provided. Cannot run council.")
        return {"error": "No modality data provided"}

    # ── Run all available agents in parallel ───────────────────────────────────
    print(f"[INFO] Running {len(parallel_branches)} agent(s) in parallel: "
          f"{list(parallel_branches.keys())}")

    parallel_runner = RunnableParallel(**parallel_branches)
    raw_results = parallel_runner.invoke({})

    # ── Tag results — skip any that are fully empty ────────────────────────────
    agent_name_map = {
        "clinical": "Clinical Data Agent (GPT-4o)",
        "genomic":  "DNA/Genomic Agent (Gemini 2.5 Flash)",
        "ct":       "CT Agent (Llama 3.3 70B)",
        "xray":     "X-Ray Agent (Qwen3 32B)",
    }

    agent_conclusions = []
    for key, text in raw_results.items():
        if text:  # skip completely empty results
            agent_conclusions.append({
                "agent": agent_name_map[key],
                "conclusion": text,
            })

    available = [c["agent"] for c in agent_conclusions if "UNAVAILABLE" not in c["conclusion"]]
    degraded  = [c["agent"] for c in agent_conclusions if "UNAVAILABLE"     in c["conclusion"]]

    print(f"\n[OK]   Available agents  : {available}")
    if degraded:
        print(f"[WARN] Degraded agents   : {degraded}")
    print()

    # Guard: if ALL agents failed, return a clean error instead of sending an
    # empty transcript to the Judge.
    if not available:
        return {
            "error": "All agents failed. Check API keys and network connectivity.",
            "agent_conclusions": agent_conclusions,
            "verdict_text": "Council could not convene — all specialist agents failed.",
            "needs_rag": False,
            "rag_query": "",
        }

    # ── Judge Agent with RAG retry loop ───────────────────────────────────────
    MAX_RAG_RETRIES = 2
    extra_rag_context = ""

    for attempt in range(MAX_RAG_RETRIES):
        print(f"[INFO] Judge Agent deliberating... (attempt {attempt + 1})")
        result = judge_agent(agent_conclusions, vector_store, rag_context=extra_rag_context)

        if not result["needs_rag"]:
            print("\n[OK] Consensus reached.\n")
            break
        else:
            print(f"[INFO] No consensus. Judge requested RAG: '{result['rag_query']}'")
            extra_rag_context = "\n\n".join(
                d.page_content for d in retrieve(result["rag_query"], vector_store)
            )
            print("[INFO] Retrieved additional evidence. Retrying...\n")

    print("=" * 60)
    print(result["verdict_text"])
    print("=" * 60)

    return result