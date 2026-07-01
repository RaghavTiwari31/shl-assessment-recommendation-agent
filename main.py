"""
main.py — Phase 2: FastAPI Application Entry Point

Endpoints:
    GET  /health  → {"status": "ok"} with HTTP 200
    POST /chat    → stateless conversation handler

All heavy initialization (catalog loading, BM25 + FAISS indexing,
embedding model loading) happens once in the startup lifespan event
so per-request latency stays well within the 30-second timeout.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging

from models import ChatRequest, ChatResponse, HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy third-party loggers during startup
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Lifespan: warm up retriever at startup (runs exactly once)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load and index the catalog on startup so the first /chat request
    doesn't pay the initialization cost.
    """
    logger.info("Starting up — loading retriever indexes...")
    # Importing retriever triggers module-level index building
    import retriever  # noqa: F401
    logger.info("Retriever ready. Importing agent...")
    import agent      # noqa: F401
    logger.info("Agent ready. Service is live.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SHL Assessment Advisor",
    description=(
        "Conversational agent for selecting SHL Individual Test Solutions. "
        "Stateless: every POST /chat call carries the full conversation history."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, summary="Readiness check")
async def health() -> HealthResponse:
    """
    Health check endpoint.
    Returns HTTP 200 with {"status": "ok"} when the service is ready.
    The evaluator allows up to 2 minutes for cold-start on this endpoint.
    """
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse, summary="Conversational assessment advisor")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless chat endpoint.

    Accepts the full conversation history on every call.
    Returns the agent's next reply, an optional structured shortlist of
    SHL assessments, and an end_of_conversation flag.
    """
    try:
        from agent import run_agent
        response = run_agent(request)
        logger.info(
            "Chat handled | turns=%d | intent_resulted_in_recs=%s | eoc=%s",
            len(request.messages),
            bool(response.recommendations),
            response.end_of_conversation,
        )
        return response
    except Exception as exc:
        logger.exception("Unhandled error in /chat: %s", exc)
        # Return a safe, schema-compliant error response instead of a 500
        return ChatResponse(
            reply="I encountered an internal error. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )


# ---------------------------------------------------------------------------
# Run directly for local development
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
