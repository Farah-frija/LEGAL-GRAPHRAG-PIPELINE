import json
import os
import re
import time
from loguru import logger
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────



GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.1-flash-lite"

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set.")

gemini = genai.Client(api_key=GEMINI_API_KEY)

# ── Schéma de réponse structuré ──────────────────────────────────────────────

_TRANSLATION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "detected_lang": types.Schema(type=types.Type.STRING, description="The detected language code: 'en' or 'ar'"),
        "en": types.Schema(type=types.Type.STRING, description="The English version of the text"),
        "ar": types.Schema(type=types.Type.STRING, description="The Arabic version of the text"),
    },
    required=["detected_lang", "en", "ar"],
)

# ── Tokenizers (Locaux pour la performance BM25) ─────────────────────────────

def tokenize_en(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric for English."""
    return re.findall(r"\w+", text.lower())

def tokenize_ar(text: str) -> list[str]:
    """Arabic tokenization — keep only Arabic characters."""
    return re.findall(r"[\u0600-\u06FF]+", text)

# ── Traduction et Détection unifiée via Gemini ───────────────────────────────

def get_both(text: str) -> dict[str, str]:
    """
    Détecte la langue et traduit en une seule requête LLM structurée.
    Retourne {"en": ..., "ar": ...}
    """
    t0 = time.perf_counter()
    logger.debug(f"Début de get_both pour le texte : {text[:50]}...")

    prompt = (
        "Analyze the following text. Detect if it is English or Arabic. "
        "Provide the text in both English and Arabic. "
        f"Text to process: '{text}'"
    )

    try:
        response = gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_TRANSLATION_SCHEMA,
                temperature=0.0,
                system_instruction=(
                    "You are a professional legal translator specialized in Omani law. "
                    "Ensure translations are precise and contextually appropriate."
                )
            ),
        )
        
        result = json.loads(response.text)
        
        elapsed = time.perf_counter() - t0
        logger.success(
            f"Traduction et détection réussies en {elapsed:.3f}s. "
            f"Langue détectée : {result['detected_lang']}"
        )
        
        return {"en": result["en"], "ar": result["ar"]}

    except Exception as e:
        logger.error(f"Erreur lors de l'appel de traduction Gemini : {e}")
        # En cas d'échec, on relève l'erreur pour que le pipeline principal soit averti
        raise