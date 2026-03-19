# translator.py - Multi-language translation using HuggingFace NLLB model (local)

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Language code mapping for NLLB model
LANG_CODES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Gujarati": "guj_Gujr",
    "Marathi": "mar_Deva",
}

LANG_LABELS = {
    "🇬🇧 English": "English",
    "🇮🇳 हिंदी (Hindi)": "Hindi",
    "🇮🇳 ગુજરાતી (Gujarati)": "Gujarati",
    "🇮🇳 मराठी (Marathi)": "Marathi",
}

MODEL_NAME = "facebook/nllb-200-distilled-600M"


@st.cache_resource
def load_translation_model():
    """Load NLLB model and tokenizer (cached so it only downloads once)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def translate_text(text, source_lang, target_lang):
    """
    Translate text from source_lang to target_lang using local NLLB model.
    
    Args:
        text: Text to translate
        source_lang: Source language name (e.g., "English")
        target_lang: Target language name (e.g., "Hindi")
    
    Returns:
        Translated text string.
    """
    if source_lang == target_lang:
        return text

    src_code = LANG_CODES.get(source_lang)
    tgt_code = LANG_CODES.get(target_lang)

    if not src_code or not tgt_code:
        return text

    tokenizer, model = load_translation_model()

    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=src_code,
        tgt_lang=tgt_code,
        max_length=512,
    )

    # Split into paragraphs and translate each
    paragraphs = text.split("\n")
    translated = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            translated.append("")
            continue
        try:
            result = translator(para)
            translated.append(result[0]["translation_text"])
        except Exception:
            translated.append(para)  # Keep original on error

    return "\n".join(translated)
