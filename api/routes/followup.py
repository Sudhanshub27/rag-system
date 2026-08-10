from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["followup"])


class FollowupRequest(BaseModel):
    last_question: str
    last_answer: str


@router.post("/followups")
async def generate_followups(req: FollowupRequest):
    """
    Generate 2-3 suggested follow-up questions based on the last Q&A exchange.
    """
    q = req.last_question.lower()

    if "summarize" in q or "explain" in q or "overview" in q:
        followups = [
            "What are the main key metrics or rules mentioned?",
            "Can you break down the most important section in detail?",
            "Are there any specific limitations or exceptions stated?",
        ]
    elif "policy" in q or "process" in q:
        followups = [
            "What are the step-by-step instructions for this process?",
            "Who is responsible for executing this procedure?",
            "What happens if there is a failure or non-compliance?",
        ]
    else:
        followups = [
            f"Can you elaborate further on {req.last_question[:25]}...?",
            "Show me the exact document page text where this is defined.",
            "Generate a Mermaid diagram illustrating this concept.",
        ]

    return {"followups": followups}
