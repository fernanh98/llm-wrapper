"""Public alias for :mod:`llm_wrapper.src.base_llm`.

The implementation lives under ``llm_wrapper/src``; this module only
re-exports it so the public import path stays short::

    from llm_wrapper.base_llm import BaseLLM
"""

from .src.base_llm import BaseLLM

__all__ = ["BaseLLM"]
