# SHL Assessment Advisor — Technical Approach Document

**Author:** SHL AI Intern Candidate  
**Assignment:** AI Intern Assessment — Conversational SHL Assessment Recommendation Agent

---

## 1. Problem Understanding & Scope

The goal is to build a stateless conversational agent that helps hiring managers select appropriate **SHL Individual Test Solutions** for a given role. The agent must handle multi-turn dialogue (up to 8 turns), return a structured JSON shortlist of 1–10 assessments per conversation, and expose two HTTP endpoints (`GET /health`, `POST /chat`) consumable by an automated evaluator.

**Hard constraints:**
- Stateless: full conversation history is sent on every `POST /chat` call; the server holds no session state.
- Scope restricted: only Individual Test Solutions — Pre-packaged Job Solutions are excluded.
- Schema strict: response must match `{reply, recommendations[{name, url, test_type}], end_of_conversation}`.
- 30-second response latency budget per turn.
- Evaluated on **Recall@10**: fraction of labeled expected assessments appearing in the agent's shortlist.

---

## 2. Architecture Overview

```
POST /chat (full history)
        │
        ▼
[Turn-Budget Guard]          ← force RECOMMEND if turns ≥ 6
        │
        ▼
[Intent Classifier]          ← single Groq call; returns one label
        │
   ┌────┴──────────────────────────────────────────────┐
   │ CLARIFY │ RECOMMEND │ COMPARE │ REFINE │ REFUSE   │
   └─────────┴─────────┴──────────┴────────┴──────────┘
        │
        ▼
[Hybrid Retriever]           ← BM25 + FAISS/dense, merged via RRF
        │
        ▼
[Groq Generator]           ← grounded reply + JSON recommendation block
        │
        ▼
[Validation Middleware]      ← strip any URL not in catalog; use catalog-authoritative names
        │
        ▼
ChatResponse
```

---

## 3. Data Engineering

**Source:** `shl_product_catalog.json` — 377 raw items.

**Filtering (scope enforcement):** Two-layer defense to exclude Pre-packaged Job Solutions:
1. **URL slug matching** — products whose URL contains `/solutions/` or `/focus-` are flagged.
2. **Name regex** — products containing "Solution" or "Focus [0-9]" in the name are cross-checked.

This reduced the working catalog from 377 → **367 Individual Test Solutions**.

**Normalization:** Empty strings in `duration`, `languages`, and `job_levels` are replaced with `"Not specified"` to prevent retrieval tokenization artifacts.

**Test-type mapping:** The catalog's `keys` field (e.g. `"Ability & Aptitude"`) is mapped to single-letter codes per the assignment schema:

| Catalog key | Code |
|---|---|
| Ability & Aptitude | A |
| Assessment Exercises | E |
| Biodata & Situational Judgment | B |
| Competencies | C |
| Development & 360 | D |
| Knowledge & Skills | K |
| Personality & Behavior | P |
| Simulations | S |

**Deduplication policy:** Near-variants (OPQ32r, OPQ32, OPQ Manager Plus) are **kept** rather than merged, because the evaluator's expected shortlists may include specific variants. Crowding is managed at the ranking stage with a per-product-family diversity constraint.

---

## 4. Hybrid Retrieval (BM25 + FAISS + RRF)

Each product is indexed as a concatenated document:
```
{name}. {description}. Keys: {keys}. Levels: {job_levels}. Duration: {duration}.
```

**BM25 (keyword):** Captures exact terminology ("OPQ", "Verify", specific technology names like "Java 8"). Built using `rank_bm25`.

**Dense (semantic):** `sentence-transformers/all-MiniLM-L6-v2` encodes all 367 documents into 384-dimensional vectors stored in a FAISS `IndexFlatIP` (inner product / cosine similarity). Handles paraphrases and synonyms ("personality test" ↔ "OPQ", "cognitive" ↔ "reasoning").

**Reciprocal Rank Fusion (RRF):** Scores from both systems are merged with `score = 1/(k + rank)` where `k=60`. RRF is parameter-free, robust to score scale differences, and consistently outperforms linear combination in information retrieval benchmarks.

**Filters:** Optional `job_level` and `keys_filter` narrow the candidate pool before ranking, with graceful fallback to full search if filters yield no results.

---

## 5. Intent Classification Router

A lightweight Groq call classifies each conversation's current state into one of five intents before any retrieval happens. This isolates pipeline logic and prevents over-querying the catalog on clarification turns.

| Intent | Trigger | Action |
|---|---|---|
| **CLARIFY** | Query too vague | Ask one targeted question (role, level, or skill) |
| **RECOMMEND** | Sufficient context | Run hybrid search → Groq generates shortlist |
| **COMPARE** | User names specific products | Catalog lookup → grounded comparison |
| **REFINE** | Adjusting a previous shortlist | Re-run recommend with updated history |
| **REFUSE** | Off-topic (salary, legal, coding) | Hard-coded rejection; zero retrieval calls |

**Turn-budget enforcement:** If `len(messages) ≥ 6`, the intent is forced to `RECOMMEND` regardless of classifier output, guaranteeing the agent delivers a shortlist within the 8-turn hard limit.

---

## 6. Anti-Hallucination Validation Middleware

After Groq generates a reply containing a JSON recommendation block, every item is cross-referenced against an in-memory URL-keyed catalog dictionary (`_CATALOG_LOOKUP`). Items are:
- **Accepted** only if their URL exactly matches a key in the clean catalog.
- **Replaced** with the catalog's authoritative `name`, `url`, and `test_type` — the model's text is not trusted for these fields.
- **Discarded** if the URL does not match any catalog entry.

If the LLM produces no valid JSON (e.g. 503 overloaded retry) the top-5 raw retrieval results are surfaced directly as a safe fallback.

This ensures the `hard-eval` constraint — "items from catalog only" — can never be violated by a model hallucination.

---

## 7. API Contract

### `GET /health`
```json
{"status": "ok"}
```
Returns HTTP 200 when the service is ready. Cold-start (index building + embedding model load) takes ~30 seconds; the evaluator allows 2 minutes.

### `POST /chat`
**Request:**
```json
{
  "messages": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
**Response:**
```json
{
  "reply": "Based on your requirements...",
  "recommendations": [
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": true
}
```
`end_of_conversation` is set to `true` when a non-empty shortlist is delivered **or** when `len(messages) ≥ 8`.

---

## 8. Evaluation: Self-Play Harness

A `evaluator.py` script drives automated evaluation:

1. **SimulatedUser** — a Groq instance loaded with a trace's `persona` and `facts` that answers the agent's clarifying questions truthfully.
2. **Conversation loop** — the simulated user and agent exchange turns until `eoc=True` or turn limit.
3. **Recall@10** — computed per trace as `|expected ∩ returned| / |expected|` with case-insensitive substring matching to tolerate minor name variants.
4. **Behavior probes** — hardcoded binary assertions:
   - Vague first turn must return 0 recommendations.
   - Off-topic queries must be refused.
   - Prompt injection must not leak system context.
   - Refinement edits must be honored in the next shortlist.
   - All returned URLs must be from the SHL domain.

---

## 9. Deployment

The service is containerized via a multi-stage `Dockerfile`. Environment variable `GROQ_API_KEY` is passed at runtime. The stateless design means any number of replicas can run behind a load balancer with no shared state required.

**Recommended platform:** Render (free tier supports persistent HTTP services with environment secrets). The `/health` endpoint satisfies Render's readiness probe.

---

## 10. Key Design Decisions & Trade-offs

| Decision | Rationale |
|---|---|
| Stateless API | Required by assignment; simplifies horizontal scaling |
| RRF over weighted sum | No calibration needed; robust across query types |
| In-memory FAISS | Eliminates external vector DB dependency; 367 docs fit comfortably in RAM |
| `llama-3.3-70b-versatile` | Fast, high-quality open-weights model; excellent for RAG |
| Validation middleware | Prevents hallucination from ever reaching the hard-eval; zero-cost safety net |
| Force-RECOMMEND at turn 6 | Guarantees 8-turn budget compliance without complex state machine |
| Keep OPQ/Excel variants | Avoids false negatives if evaluator expects a specific variant |
