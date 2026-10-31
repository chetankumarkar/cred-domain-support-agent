"""Seeded synthetic loan applications and status lookup tool."""
from __future__ import annotations
import json, random
from collections import Counter
from pathlib import Path

SEED = 20261031
N = 50
AMOUNT_RANGE = (50_000, 5_000_000)
CATEGORIES = ["Personal Loan", "Home Loan", "Auto Loan", "Education Loan", "Business Loan"]
STATUSES = ["Submitted", "Under Review", "Approved", "Rejected", "Disbursed"]
FRAUD_PROBABILITY = 0.20
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "loan_applications.json"

def generate_records(seed: int = SEED, n: int = N) -> list[dict]:
    if n < 40 or n < len(CATEGORIES) * 3:
        raise ValueError("n must be at least 40 and allow 3 records per category")
    rng = random.Random(seed)
    categories = [CATEGORIES[i % len(CATEGORIES)] for i in range(n)]
    statuses = [STATUSES[i % len(STATUSES)] for i in range(n)]
    rng.shuffle(categories); rng.shuffle(statuses)
    records = [{
        "record_id": f"CRD-{i+1:04d}", "category": categories[i], "status": statuses[i],
        "loan_amount_inr": rng.randrange(AMOUNT_RANGE[0] // 1000, AMOUNT_RANGE[1] // 1000 + 1) * 1000,
        "days_since_created": rng.randint(0, 30),
        "flagged_for_fraud_review": rng.random() < FRAUD_PROBABILITY,
    } for i in range(n)]
    validate_records(records)
    return records

def validate_records(records: list[dict]) -> None:
    if len(records) < 40: raise ValueError("at least 40 records are required")
    required = {"record_id", "category", "status", "loan_amount_inr", "days_since_created", "flagged_for_fraud_review"}
    for row in records:
        if set(row) != required: raise ValueError(f"invalid fields: {row}")
        if row["category"] not in CATEGORIES or row["status"] not in STATUSES: raise ValueError("unknown category/status")
        if not AMOUNT_RANGE[0] <= row["loan_amount_inr"] <= AMOUNT_RANGE[1]: raise ValueError("amount out of range")
        if not 0 <= row["days_since_created"] <= 30: raise ValueError("days out of range")
    cats, stats = Counter(x["category"] for x in records), Counter(x["status"] for x in records)
    if any(cats[x] < 3 for x in CATEGORIES): raise ValueError(f"category coverage failed: {cats}")
    if any(stats[x] < 1 for x in STATUSES): raise ValueError(f"status coverage failed: {stats}")
    pct = sum(x["flagged_for_fraud_review"] for x in records) / len(records)
    if not .10 <= pct <= .30: raise ValueError(f"fraud percentage {pct:.1%} outside 10-30%; choose a new seed/weight")

def load_records() -> list[dict]:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        rows = generate_records()
        DATA_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return rows
    rows = json.loads(DATA_PATH.read_text(encoding="utf-8")); validate_records(rows); return rows

RECORDS = load_records()
def escalation_score(record: dict) -> float:
    return round(.65 * int(record["flagged_for_fraud_review"]) + .35 * (1 - record["days_since_created"] / 30), 4)

def check_loan_application_status(record_id: str) -> dict:
    """Return only application status, amount, and risk routing score."""
    row = next((r for r in RECORDS if r["record_id"].casefold() == record_id.strip().casefold()), None)
    if row is None: return {"found": False, "record_id": record_id, "message": "No application matched that reference."}
    return {"found": True, "record_id": row["record_id"], "category": row["category"], "status": row["status"],
            "loan_amount_inr": row["loan_amount_inr"], "escalation_score": escalation_score(row),
            "escalate": escalation_score(row) >= .65}

def report() -> dict:
    return {"count": len(RECORDS), "category_counts": dict(Counter(x["category"] for x in RECORDS)),
            "status_counts": dict(Counter(x["status"] for x in RECORDS)),
            "fraud_review_percent": round(100 * sum(x["flagged_for_fraud_review"] for x in RECORDS) / len(RECORDS), 2)}

if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(RECORDS, indent=2), encoding="utf-8")
    print(json.dumps(report(), indent=2))
