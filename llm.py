import os
import google.generativeai as genai

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDOn4ct8Xj42_bd6v0X58hvTPAarlswOJc")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def llm_generate(prompt: str) -> str:
    """
    Génère du texte via Gemini (clé Google AI Studio).

    Si vous obtenez une erreur "model not found", changez GEMINI_MODEL
    ou listez les modèles disponibles avec genai.list_models().
    """
    if LLM_PROVIDER != "gemini":
        raise RuntimeError("LLM_PROVIDER invalide. Utilisez 'gemini'.")

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY manquant. Ajoutez-le dans .env (local) ou dans Secrets (Streamlit Cloud)."
        )

    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=500,
        ),
    )

    resp = model.generate_content(prompt)
    return (resp.text or "").strip()
