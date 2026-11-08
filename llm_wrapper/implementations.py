"""Public alias for :mod:`llm_wrapper.src.implementations`.

The implementation lives under ``llm_wrapper/src``; this module only
re-exports it so the public import path stays short::

    from llm_wrapper.implementations import create_llm

Importing this module registers the built-in providers, exactly like importing
``llm_wrapper.src.implementations`` does.
"""

from .src.implementations import LLMManual, LLMOpenai, create_llm

__all__ = ["LLMManual", "LLMOpenai", "create_llm"]
