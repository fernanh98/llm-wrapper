from typing import Generic, Optional, TypeVar
from pydantic import BaseModel, Field
from enum import Enum


StructuredT = TypeVar("StructuredT", bound=BaseModel)

class Roles(str, Enum):
    LLM = "assistant"
    USER = "user"
    SYSTEM = "system"
    TOOL = "tool" # TODO: review key value


class ToolCall(BaseModel):
    name: str = Field(
        description="Name of the tool"
    )
    arguments: dict = Field(
        description="Arguments of the tool"
    )


class Message(BaseModel, Generic[StructuredT]):
    role: Roles
    content: Optional[str] = Field(
        default=None,
        description="Content of the response in plain text format"
    )
    structured_content: Optional[StructuredT] = Field(
        default=None,
        description="Parsed structured output, when a Pydantic response format is requested"
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool calls requested by the model"
    )
