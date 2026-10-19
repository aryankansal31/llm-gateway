"""The subset of the OpenAI Chat Completions API the gateway accepts and returns."""

import time
import uuid
from typing import Literal

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[dict] | None = None

    def text(self) -> str:
        """Plain text of the message. Content parts are joined; non-text parts are rejected."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        parts = []
        for part in self.content:
            if part.get("type") != "text":
                raise ValueError("unsupported_content")
            parts.append(part.get("text", ""))
        return "\n".join(parts)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = 1.0
    max_tokens: int | None = None
    response_format: dict | None = None
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str | None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage


def error_response(
    status: int, message: str, type_: str, code: str | None = None, headers: dict | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "code": code}},
        headers=headers,
    )
