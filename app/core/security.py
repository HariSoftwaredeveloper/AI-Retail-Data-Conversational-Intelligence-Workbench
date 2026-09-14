"""Security, validation, and privacy protection utilities.
"""
import re
from pathlib import Path
from typing import Any, Dict, List
from app.core.config import DATA_DIR, ARTIFACTS_DIR


EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}")
CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"print\s+(secret|password|api[-_]?key)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
    re.compile(r"bypass\s+all\s+rules", re.IGNORECASE),
]


def sanitize_filename(filename: str) -> str:
    """Strip dangerous path traversal characters and return safe basename."""
    safe_name = Path(filename).name
    # Keep only alphanumeric, underscores, hyphens, and single dots
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", safe_name)
    if not safe_name.endswith(".csv"):
        safe_name = f"{safe_name}.csv"
    return safe_name


def resolve_safe_path(base_dir: Path, filename: str) -> Path:
    """Ensure resolved path is strictly within base_dir."""
    safe_name = sanitize_filename(filename)
    target = (base_dir / safe_name).resolve()
    if not str(target).startswith(str(base_dir.resolve())):
        raise ValueError(f"Illegal path traversal attempt: {filename}")
    return target


def mask_pii_string(val: str) -> str:
    """Mask email, phone numbers, and card numbers."""
    if not isinstance(val, str):
        return val
    masked = EMAIL_REGEX.sub("[EMAIL_MASKED]", val)
    masked = CARD_REGEX.sub("[CARD_MASKED]", masked)
    masked = PHONE_REGEX.sub("[PHONE_MASKED]", masked)
    return masked


def mask_pii_in_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mask PII in sample records before any external exposure."""
    sanitized = []
    sensitive_columns = {"email", "phone", "phone_number", "ssn", "credit_card", "password", "address"}
    for record in records:
        row = {}
        for k, v in record.items():
            if str(k).lower() in sensitive_columns:
                row[k] = "[REDACTED]"
            elif isinstance(v, str):
                row[k] = mask_pii_string(v)
            else:
                row[k] = v
        sanitized.append(row)
    return sanitized


def check_for_prompt_injection(text: str) -> bool:
    """Check if input text attempts prompt injection."""
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False
