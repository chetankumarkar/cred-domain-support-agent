"""Deterministic input and output safety controls for mock-first operation."""
from __future__ import annotations
import re

PATTERNS = [
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.I)),
    ("AADHAAR", re.compile(r"(?<!\d)(?:\d{4}[ -]?){2}\d{4}(?!\d)")),
    ("BANK_ACCOUNT", re.compile(r"\b\d{9,18}\b")),
]
INJECTION = re.compile(r"(ignore (all )?(previous|prior) instructions|reveal (the )?(system prompt|secrets?)|bypass (the )?(policy|guardrail)|act as (the )?system)", re.I)

class GuardrailError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message); self.code = code

def mask_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for name, pattern in PATTERNS:
        text, n = pattern.subn(lambda m: "[MASKED_" + name + "]", text)
        if n: found.append(name)
    return text, found

def validate_input(text: str) -> tuple[str, dict]:
    masked, fields = mask_pii(text)
    if INJECTION.search(masked): raise GuardrailError("PROMPT_INJECTION", "Request rejected by the prompt-injection guardrail.")
    return masked, {"pii_fields_masked": fields, "injection_detected": False}

def groundedness_check(answer: str, context: str, abstained: bool = False) -> str:
    if abstained: return "I don't know based on the available policy documents. Please ask an authorized support representative."
    # Mock groundedness policy: every non-fallback factual answer carries a retrieved source excerpt.
    if not context.strip() or not answer.strip():
        raise GuardrailError("UNGROUNDED_OUTPUT", "The answer is not supported by retrieved context.")
    return answer
