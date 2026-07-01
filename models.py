"""
models.py — Phase 2: Pydantic request / response schemas

These models are the single source of truth for the API contract.
The assignment schema is non-negotiable — deviating breaks the evaluator.
"""

from pydantic import BaseModel, Field
from typing import Optional


class Message(BaseModel):
    """A single turn in the conversation history."""
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The text content of the turn")


class ChatRequest(BaseModel):
    """POST /chat request body — full stateless conversation history."""
    messages: list[Message] = Field(
        ...,
        min_length=1,
        description="Full conversation history including the latest user message",
    )


class Recommendation(BaseModel):
    """A single assessment recommendation item."""
    name: str = Field(..., description="Official assessment name from the SHL catalog")
    url: str = Field(..., description="Canonical SHL catalog URL")
    test_type: str = Field(..., description="Primary test type abbreviation (e.g. K, P, A)")


class ChatResponse(BaseModel):
    """POST /chat response — fixed schema required by the assignment evaluator."""
    reply: str = Field(..., description="Agent's conversational reply text")
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="1-10 assessments when agent has a shortlist, empty otherwise",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the task complete",
    )


class HealthResponse(BaseModel):
    """GET /health response."""
    status: str = "ok"
