"""Pydantic models for OpenAI-compatible chat and tool calling."""
from __future__ import annotations
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from .config import DEFAULT_MODEL


class FunctionCall(BaseModel):
    name: str
    arguments: str = "{}"


class AssistantToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[dict], None] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[AssistantToolCall]] = None


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: dict = Field(default_factory=lambda: {"type": "object"})
    strict: Optional[bool] = None


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = False
    conversation_id: Optional[str] = None
    thinking: bool = False
    search: bool = False
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, dict]] = None
    parallel_tool_calls: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    user: Optional[str] = None
