import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PROGNOSIS AGENT (Primary) — GPT-4o
# Role: Initial TB likelihood assessment
# Provider: OpenAI via GitHub Models (free)
# ─────────────────────────────────────────────
def test_prognosis_primary():
    from openai import OpenAI
    try:
        client = OpenAI(
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url="https://models.inference.ai.azure.com"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Respond with OK"}]
        )
        print("✅ Prognosis Agent - Primary (GPT-4o via GitHub):", response.choices[0].message.content)
    except Exception as e:
        print("❌ Prognosis Agent - Primary (GPT-4o) FAILED:", e)


# ─────────────────────────────────────────────
# PROGNOSIS AGENT (Secondary) — Gemini 2.5 Flash
# Role: Independent second TB assessment
# Provider: Google AI Studio (free)
# ─────────────────────────────────────────────
def test_prognosis_secondary():
    from langchain_google_genai import ChatGoogleGenerativeAI
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        response = llm.invoke("Respond with OK")
        print("✅ Prognosis Agent - Secondary (Gemini 2.5 Flash):", response.content)
    except Exception as e:
        print("❌ Prognosis Agent - Secondary (Gemini 2.5 Flash) FAILED:", e)


# ─────────────────────────────────────────────
# DEVIL'S ADVOCATE AGENT (Primary) — Llama 3.3 70B
# Role: Challenge prognosis, propose alternatives
# Provider: Groq (free)
# ─────────────────────────────────────────────
def test_devils_advocate_primary():
    from groq import Groq
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Respond with OK"}]
        )
        print("✅ Devil's Advocate Agent - Primary (Llama 3.3 70B):", response.choices[0].message.content)
    except Exception as e:
        print("❌ Devil's Advocate Agent - Primary (Llama 3.3 70B) FAILED:", e)


# ─────────────────────────────────────────────
# DEVIL'S ADVOCATE AGENT (Secondary) — Qwen3 32B
# Role: Additional challenge, flag insufficient evidence
# Provider: Groq (free)
# ─────────────────────────────────────────────
def test_devils_advocate_secondary():
    from groq import Groq
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": "Respond with OK"}]
        )
        print("✅ Devil's Advocate Agent - Secondary (Qwen3 32B):", response.choices[0].message.content)
    except Exception as e:
        print("❌ Devil's Advocate Agent - Secondary (Qwen3 32B) FAILED:", e)


# ─────────────────────────────────────────────
# JUDGE / SYNTHESIZER AGENT — GPT-OSS 120B
# Role: Read full debate, issue final verdict + confidence score
# Provider: Groq (free)
# ─────────────────────────────────────────────
def test_judge_synthesizer():
    from groq import Groq
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Respond with OK"}]
        )
        print("✅ Judge/Synthesizer Agent (GPT-OSS 120B):", response.choices[0].message.content)
    except Exception as e:
        print("❌ Judge/Synthesizer Agent (GPT-OSS 120B) FAILED:", e)


if __name__ == "__main__":
    print("\n--- TB-Guard Council of AI — Connection Test ---\n")
    test_prognosis_primary()
    test_prognosis_secondary()
    test_devils_advocate_primary()
    test_devils_advocate_secondary()
    test_judge_synthesizer()
    print("\n--- All agents tested ---\n")
