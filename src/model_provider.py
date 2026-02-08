"""LLM provider factory with automatic fallback from Google Gemini to Groq."""

import os

GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"


def create_llm(model_name: str = None, temperature: float = 0.0):
    """Create an LLM instance using GOOGLE_API_KEY (primary) or GROQ_API_KEY (fallback).

    Args:
        model_name: Model name override. If None, auto-selects based on available API key.
        temperature: Temperature for the LLM.

    Returns:
        A LangChain chat model instance.

    Raises:
        ValueError: If neither GOOGLE_API_KEY nor GROQ_API_KEY is set.
    """
    google_key = os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or GEMINI_DEFAULT_MODEL,
            temperature=temperature,
            google_api_key=google_key,
        )
    elif groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name or GROQ_DEFAULT_MODEL,
            temperature=temperature,
            api_key=groq_key,
        )
    else:
        raise ValueError(
            "No LLM API key found. Set GOOGLE_API_KEY or GROQ_API_KEY in your .env file."
        )
