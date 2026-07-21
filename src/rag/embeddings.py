"""Embedding model configuration.

This is intentionally the *only* place that knows how text gets turned
into vectors. To swap the embedding model later, change EMBEDDING_MODEL
in your .env, or replace the function below with a different Chroma
embedding function (OpenAI, Cohere, a local model, etc.).
"""
from __future__ import annotations
import dotenv

from chromadb.utils import embedding_functions

from .config import settings
dotenv.load_dotenv()


def get_embedding_function():
    """Return the embedding function Chroma uses to vectorize text."""
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=dotenv.get_key(dotenv.find_dotenv(), "OPENROUTER_API_KEY"),
        api_base=dotenv.get_key(dotenv.find_dotenv(), "OPENROUTER_API_BASE_URL"),
        model_name=settings.embedding_model
    )
