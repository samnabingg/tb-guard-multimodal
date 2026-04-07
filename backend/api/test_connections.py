import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CLINICAL AGENT — GPT-4o
# Role: Analyze clinical TB indicators from patient records
# Provider: OpenAI via GitHub Models (free)
# ─────────────────────────────────────────────
def test_clinical_agent():
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
        print("✅ Clinical Agent (GPT-4o via GitHub):", response.choices[0].message.content)
    except Exception as e:
        print("❌ Clinical Agent (GPT-4o) FAILED:", e)


# ─────────────────────────────────────────────
# GENOMIC AGENT — Gemini 2.5 Flash
# Role: Interpret genomic mutations and resistance markers
# Provider: Google AI Studio (free)
# ─────────────────────────────────────────────
def test_genomic_agent():
    from langchain_google_genai import ChatGoogleGenerativeAI
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        response = llm.invoke("Respond with OK")
        print("✅ Genomic Agent (Gemini 2.5 Flash):", response.content)
    except Exception as e:
        print("❌ Genomic Agent (Gemini 2.5 Flash) FAILED:", e)


# ─────────────────────────────────────────────
# CT AGENT — Llama 3.3 70B
# Role: Analyze CT imaging findings for TB patterns
# Provider: Groq (free)
# ─────────────────────────────────────────────
def test_ct_agent():
    from groq import Groq
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Respond with OK"}]
        )
        print("✅ CT Agent (Llama 3.3 70B):", response.choices[0].message.content)
    except Exception as e:
        print("❌ CT Agent (Llama 3.3 70B) FAILED:", e)


# ─────────────────────────────────────────────
# X-RAY AGENT — Qwen3 32B
# Role: Analyze chest X-ray findings for TB evidence
# Provider: Groq (free)
# ─────────────────────────────────────────────
def test_xray_agent():
    from groq import Groq
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": "Respond with OK"}]
        )
        print("✅ X-Ray Agent (Qwen3 32B):", response.choices[0].message.content)
    except Exception as e:
        print("❌ X-Ray Agent (Qwen3 32B) FAILED:", e)


# ─────────────────────────────────────────────
# INTEGRATION AGENT — GPT-OSS 120B
# Role: Integrate all modality conclusions into final TB verdict
# Provider: Groq (free)
# ─────────────────────────────────────────────
def test_integration_agent():
    from groq import Groq
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Respond with OK"}]
        )
        print("✅ Integration Agent (GPT-OSS 120B):", response.choices[0].message.content)
    except Exception as e:
        print("❌ Integration Agent (GPT-OSS 120B) FAILED:", e)


if __name__ == "__main__":
    print("\n--- TB-Guard Council of AI — Connection Test ---\n")
    test_clinical_agent()
    test_genomic_agent()
    test_ct_agent()
    test_xray_agent()
    test_integration_agent()
    print("\n--- All agents tested ---\n")
