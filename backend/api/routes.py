
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.orchestrator.orchestrator import Orchestrator


logger = logging.getLogger(__name__)

router = APIRouter()

orchestrator = Orchestrator()


# ============================================================
# Request schema
# ============================================================

class QueryRequest(BaseModel):
    query: Optional[str] = None
    question: Optional[str] = None
    external_context: Optional[str] = None

    def get_query_text(self) -> str:
        """
        Support both query and question for backward compatibility.
        """
        text = self.query or self.question

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "A non-empty query or question is required."
            )

        return text.strip()


# ============================================================
# Response schema
# ============================================================

class QueryResponse(BaseModel):
    answer: str

    claims: list[dict[str, Any]] = Field(
        default_factory=list
    )

    hallucination_score: float = Field(
        ge=0.0,
        le=1.0
    )

    relevance_score: float = Field(
        ge=0.0,
        le=1.0
    )


# ============================================================
# Medical query endpoint
# ============================================================

@router.post(
    "/ask",
    response_model=QueryResponse,
    summary="Ask ParmanaAI a medical question",
    description=(
        "Generates an evidence-grounded response and returns "
        "claim-level NLI and numerical verification information."
    ),
)
def ask_medical_question(
    request: QueryRequest,
) -> QueryResponse:

    try:
        query_text = request.get_query_text()

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    try:
        response = orchestrator.run_pipeline(
            query_text,
            external_context=request.external_context,
        )

        if not isinstance(response, dict):
            logger.error(
                "Orchestrator returned an invalid response type."
            )

            raise HTTPException(
                status_code=500,
                detail="Internal response processing error.",
            )

        # Validate the complete response before returning it.
        validated_response = QueryResponse.model_validate(
            response
        )

        return validated_response

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Medical query processing failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process the medical query. "
                "Please try again later."
            ),
        )
