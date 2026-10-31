# Cred (Banking & FinTech) capstone completed: Domain Support Agent

This repository contains a production-minded, locally runnable reference implementation of the Cred loan-support capstone. It answers policy questions from an authored knowledge base, looks up fabricated application records, maintains session history, masks fixed-format PII, rejects injection and oversized requests, reviews drafts, logs requests safely, and exposes HTTP and WebSocket APIs.

## Reproduce the dataset

- Seed: `20261031`
- Categories (equal weights): Personal Loan, Home Loan, Auto Loan, Education Loan, Business Loan
- Statuses (equal weights): Submitted, Under Review, Approved, Rejected, Disbursed
- Amount range: INR 50,000–5,000,000. This range covers small unsecured loans through larger secured and business loans in a compact synthetic portfolio.
- Application count: 50
- Fraud-review probability: 20% (seeded; records are never manually adjusted)

Run `python -m cred_agent.dataset` to regenerate and validate `data/loan_applications.json` and print the category, status, and fraud-review counts.

## Quick start

Python 3.11+ is recommended. Install dependencies with `pip install -r requirements.txt`, then run `python -m cred_agent.demo` for reproducible transcripts and `uvicorn cred_agent.api:app --host 127.0.0.1 --port 8000` for the API. With dependencies installed, the demo runs the CrewAI `.kickoff()` and AutoGen round-robin review paths and records their outputs.

The default is `MOCK_LLM=1`; it requires no API key or paid model account. The API sets `USE_CREWAI_CREW=1` so requests pass through the same CrewAI crew exercised in the demo. `CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`, and `CREWAI_TRACING_ENABLED=false` are set before importing/starting CrewAI to prevent telemetry or trace-upload prompts. Embeddings use a free local SentenceTransformers model, and ChromaDB stores separate fixed-window and sentence-based collections. On the first setup, the model must be downloaded once (free, but uses internet data) or already present locally; then run `python -m cred_agent.index` to build both collections and set `USE_VECTOR_STORE=1` to query them. Deterministic lexical retrieval remains available for offline demos. Model inference and vector operations run locally after setup.

## Design and evidence

The repository is organized into dataset generation, knowledge base, retrieval, orchestration, safety, review, evaluation, API, and audit logs. `evidence/mock_transcripts.jsonl` records the task demonstrations: retrieval and lookup tool calls, multi-turn and fresh-session memory, PII and injection rejection, groundedness fallback, both reviewer outcomes, budget rejection, cache hit counters, retrieval comparisons, calibrated similarity examples, and all 15 evaluation rows plus averages. Rerun `python -m cred_agent.demo` after code changes to refresh this evidence.

Similarity calibration is computed using a deterministic local token-vector cosine and query-term coverage score in `cred_agent.rag` so that MOCK_LLM demonstrations are reproducible without model downloads and concise natural questions still match relevant policy text. Four in-scope probes measured 0.7518, 0.7278, 0.8175, and 0.6591; three unrelated probes measured 0.0224, 0.2124, and 0.0000. The tested midpoint threshold is 0.4358, between the observed clusters. For the production retrieval path, the project also implements local SentenceTransformers embeddings and ChromaDB indexing. The lexical score is an explicit MOCK_LLM evaluation aid, not a claim that it equals transformer embeddings; recalibrate for any alternative embedding model.

The fixed-window strategy uses 500-character chunks with 80-character overlap. The sentence strategy groups up to three sentences per chunk. Retrieval comparison uses parent-document relevance labels and deduplicates parent IDs before scoring. In the five-query mock comparison, fixed-window averages precision 0.210 and recall 1.000; sentence-based averages precision 0.257 and recall 1.000. I recommend sentence chunks because they preserve the same measured recall while improving precision by 0.047; each per-query arithmetic row is in `evidence/mock_transcripts.jsonl`.

The escalation score is `0.65 * fraud_flag + 0.35 * (1 - days_since_created / 30)`. Escalate at `>= 0.65`: with this formula every fraud-flagged record meets the cutoff and every unflagged record remains below 0.35. The generated cohort has 24% fraud-review flags, inside the required 10–30% band, so the threshold routes that empirically observed flagged minority for human review. Only the Lookup Agent receives the status lookup tool.

## Governance

Risk classification: **High**. The agent handles financial application status and policy, and incorrect or exposed data could affect members and create regulatory or operational harm. It is decision support only: it does not approve, reject, price, or disburse loans. Human staff retain authority over consequential actions. Free-text applicant names and income amounts are outside the specified deterministic masker; all examples are fabricated. PAN, Aadhaar, and account numbers are masked before model input and before logging.

The request budget is capped by input characters and estimated tokens before orchestration. The cache is in-memory and keyed by normalized query. Reviewer output is a structured verdict. For real deployment, add authentication, authorization, rate limiting, durable session storage, secrets management, and institution-approved policy content before exposing the service publicly.

## Project map

- `cred_agent/dataset.py`: seeded generator, validation, lookup, escalation score
- `knowledge_base/`: 14 original policy articles
- `cred_agent/rag.py`: chunkers, local embeddings/Chroma index, retrieval, grounded answer, calibration and scoring
- `cred_agent/crew.py`: mock-capable support orchestration, CrewAI crew, schema, memory, tool ownership, cache, budget
- `cred_agent/guardrails.py`: input masking and injection checks, output groundedness check
- `cred_agent/review.py`: deterministic reviewer and optional AutoGen RoundRobinGroupChat adapter
- `cred_agent/evaluation.py`: 15-case four-dimension MOCK_LLM evaluation
- `cred_agent/api.py`: `/ask`, `/add-document`, and `/ws/chat`
- `evidence/`: reproducible transcript and JSONL request log

## Run

```bash
python -m pip install -r requirements.txt
python -m cred_agent.dataset
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
python -m cred_agent.index
python -m cred_agent.demo
uvicorn cred_agent.api:app --host 127.0.0.1 --port 8000
```

The SentenceTransformers line downloads the embedding model once from Hugging Face if it is not cached; it is free and uses internet data. Subsequent indexing and retrieval use the local copy.

All samples and records are synthetic. This project does not use or request real member information.
