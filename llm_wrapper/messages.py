"""Public alias for :mod:`llm_wrapper.src.messages`.

The implementation lives under ``llm_wrapper/src``; this module only
re-exports it so the public import path stays short::

    from llm_wrapper.messages import Message, Roles, ToolCall
"""

from .src.messages import Message, Roles, StructuredT, ToolCall

__all__ = ["Message", "Roles", "StructuredT", "ToolCall"]
