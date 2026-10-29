"""Keep review metadata separate from the published answer text."""
from pydantic_models import VerificationResult


def publish_reviewed_answer(answer, verification):
    if isinstance(verification, dict):verification = VerificationResult.model_validate(verification)
    # Never silently reuse an assessment for different prose.
    if verification.checked_answer != answer:
        return answer, VerificationResult(status='UNAVAILABLE',checked_answer=answer,
            review_mode=verification.review_mode,warning='Сохранённая оценка относится к другому тексту.')
    return answer, verification
