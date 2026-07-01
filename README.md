# SHL Assessment Advisor

A stateless conversational AI agent for selecting SHL Individual Test Solutions.  
Developed as part of the SHL AI Intern Assignment.

---

## Live Demo
- **Live API Base URL:** [https://shl-assessment-raghav-tiwari.onrender.com](https://shl-assessment-raghav-tiwari.onrender.com)
- **Interactive Swagger Docs:** [https://shl-assessment-raghav-tiwari.onrender.com/docs](https://shl-assessment-raghav-tiwari.onrender.com/docs)

> 💡 **Note:** The service is hosted on Render's free tier and may take 30–60 seconds to wake up after periods of inactivity.

---

## Overview
The goal of this project is to provide a conversational interface for recruiters to discover the best SHL assessments for their hiring needs. It acts as an expert consultant, classifying user intents, executing hybrid searches against the SHL catalog, and returning grounded, strictly validated JSON recommendations within a guaranteed 8-turn conversation budget.

---

## Features
- **Stateless RAG Architecture:** Fully stateless design allowing for infinite horizontal scaling and robust API performance.
- **Intent-Driven Routing:** A 5-intent router (Clarify, Recommend, Compare, Refine, Refuse) prevents hallucination and ensures focused, relevant answers.
- **Hybrid Search Engine:** Combines BM25 keyword search with FAISS semantic search using Reciprocal Rank Fusion (RRF) for high precision retrieval.
- **Zero Hallucination Middleware:** Every generated assessment is strictly cross-referenced against the SHL catalog before being returned to the user.
- **Self-Play Evaluator:** A fully automated evaluation harness (`test_scripts/evaluator.py`) to grade the agent using Recall@10 against simulated user traces.

---

## Tech Stack
- **Backend Framework:** FastAPI (Python 3.11)
- **LLM Provider:** Groq (`llama-3.3-70b-versatile`) — *The application uses Groq-hosted Llama 3.3 70B for fast, accurate inference.*
- **Vector Search:** FAISS & Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Keyword Search:** Rank-BM25
- **Deployment:** Render (Native Python) + Docker support

---

## Evaluation
The project includes an automated evaluation harness that measures retrieval quality (Recall@10) and validates conversational behavior through simulated user traces.

---

## Project Structure

```
SHL_Project/
├── shl_product_catalog.json   # Raw SHL product catalog (input data)
├── data_prep.py               # Catalog filtering, normalization, test-type mapping
├── retriever.py               # BM25 + FAISS hybrid search with RRF
├── models.py                  # Pydantic request/response schemas
├── agent.py                   # Intent router + Groq pipeline + validation middleware
├── main.py                    # FastAPI app (GET /health, POST /chat)
├── test_scripts/              # Evaluation and testing suite
│   ├── evaluator.py           # Self-play evaluation harness (Recall@10 + behavior probes)
│   ├── traces/                # Evaluation trace JSON files
│   ├── smoke_test.py          # Quick 5-intent API smoke test
│   └── test_recall.py         # Unit test for metric logic
├── docs/                      # Project documentation
│   └── approach_document.md   # 2-page technical approach (assignment deliverable)
├── requirements.txt           # Python dependencies
├── render.yaml                # One-click Render deployment config
└── .env.example               # Environment variable template
```

---

## Getting Started

### Step 1 — Prerequisites

- Python **3.11 or later** ([python.org/downloads](https://www.python.org/downloads/))
- A free **Groq API key** from [console.groq.com/keys](https://console.groq.com/keys)

---

### Step 2 — Create a virtual environment

Open a terminal inside the `SHL_Project` folder and run:

```powershell
python -m venv venv
```

Then activate it:

```powershell
# Windows PowerShell
.\venv\Scripts\activate

# Windows Command Prompt
.\venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

You should see `(venv)` appear at the start of your prompt.

---

### Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

> ⏳ This takes **2–3 minutes** the first time. PyTorch and the sentence-transformers model are large packages.

---

### Step 4 — Set your API key

```powershell
# Windows PowerShell
copy .env.example .env
```

Open the `.env` file and replace `your_groq_api_key_here` with your actual key:

```env
GROQ_API_KEY=gsk_...your_actual_key...
```

Save the file.

---

### Step 5 — Start the server

```powershell
.\venv\Scripts\python main.py
```

You will see output like this:

```
[data_prep] Loaded 377 raw items.
[data_prep] Filtered out 10 Job Solution(s): ...
[data_prep] Clean catalog size: 367 Individual Test Solutions.
[retriever] Encoding catalog documents for FAISS...
[retriever] Indexes ready. BM25 + FAISS (367 docs, dim=384).
Agent ready. Service is live.
INFO: Uvicorn running on http://0.0.0.0:8000
```

> ⏳ **Cold-start takes ~30 seconds** while the embedding model loads. The server is ready when you see "Service is live."

---

### Step 6 — Test it works

Open a **second terminal** (keep the server running in the first), activate the venv, and run:

```powershell
.\venv\Scripts\python test_scripts/smoke_test.py
```

Expected output:

```
[CLARIFY]
  reply: What role or job title are you hiring for?
  recs: 0 | eoc: False

[RECOMMEND]
  reply: For your mid-level Java backend developer role...
  recs: 3 | eoc: True
    - Core Java (Advanced Level) (New) | K
    - Java 8 (New) | K
    - Java Design Patterns (New) | K
    ...

[REFUSE]
  reply: I'm here to help with SHL assessment selection only...
  recs: 0 | eoc: False
```

---

### Step 7 — (Optional) Run the evaluation harness

With the server still running in terminal 1:

```powershell
# Full evaluation (all traces + behavior probes)
.\venv\Scripts\python test_scripts/evaluator.py

# Faster — skip behavior probes
.\venv\Scripts\python test_scripts/evaluator.py --skip-probes

# Quiet mode — no per-turn conversation output
.\venv\Scripts\python test_scripts/evaluator.py --quiet
```

Results are saved to `eval_report.json`.

---

## API Reference

### `GET /health`

```
GET http://localhost:8000/health
```

Response:
```json
{"status": "ok"}
```

---

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user",      "content": "I need to hire a Java developer"},
    {"role": "assistant", "content": "What seniority level are you hiring for?"},
    {"role": "user",      "content": "Mid-level, around 4 years of experience"}
  ]
}
```

**Response:**
```json
{
  "reply": "Based on your requirements, I recommend the following assessments...",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": true
}
```

**`test_type` codes:**

| Code | Category |
|---|---|
| A | Ability & Aptitude |
| B | Biodata & Situational Judgment |
| C | Competencies |
| D | Development & 360 |
| E | Assessment Exercises |
| K | Knowledge & Skills |
| P | Personality & Behavior |
| S | Simulations |

> Interactive API docs available at: 
> - **Local:** `http://localhost:8000/docs`
> - **Production:** `https://shl-assessment-raghav-tiwari.onrender.com/docs`

---

## Step-by-Step: Deploying to Render (no Docker needed)

Render supports native Python apps — no Docker required.

### Step 1 — Push your project to GitHub

Clone the repository to your local machine:

```bash
git clone https://github.com/RaghavTiwari31/shl-assessment-recommendation-agent.git
cd shl-assessment-recommendation-agent
```

> **Important:** Make sure `.env` is in your `.gitignore` — never commit your API key.

Add a `.gitignore` if you don't have one:

```
.env
venv/
__pycache__/
*.pyc
eval_report.json
SHL_AI_Intern_Assignment.pdf
project_findings_and_flow.txt
smoke_test_logs.json
```

### Step 2 — Create a new Web Service on Render

1. Go to [render.com](https://render.com) → **New +** → **Web Service**
2. Connect your GitHub repo
3. Render will auto-detect `render.yaml` and fill in the settings

If Render doesn't auto-detect, set manually:
- **Runtime:** Python 3
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Step 3 — Add your API key

In the Render dashboard:
1. Go to your service → **Environment**
2. Add: `GROQ_API_KEY` = `<your key>`
3. Click **Save Changes**

### Step 4 — Deploy

Click **Deploy**. Render builds and starts the service.  
Your live URL will be: `https://shl-assessment-raghav-tiwari.onrender.com`

---

## Docker Deployment (Optional)

The project includes a multi-stage `Dockerfile` optimized for small footprint deployments. While the Render deployment uses the native Python runtime, you can easily containerize the app for local execution or other cloud providers:

```bash
docker build -t shl-advisor .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here shl-advisor
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY not found` | Make sure `.env` file exists and has the key set |
| `429 Too Many Requests` | Free-tier Groq quota hit. Wait for reset |
| `503 UNAVAILABLE` | Groq model temporarily overloaded. The agent auto-retries (5s, 10s, 20s) |
| Server doesn't start | Check Python version is 3.11+ with `python --version` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the activated venv |
| Slow first response | Normal — FAISS index builds in memory on startup (~30 seconds) |

---

## Architecture (Quick Reference)

```
POST /chat (full conversation history)
        │
        ▼
[Turn-Budget Guard]     → force RECOMMEND if turns ≥ 6
        │
        ▼
[Intent Classifier]     → CLARIFY | RECOMMEND | COMPARE | REFINE | REFUSE
        │
        ▼
[Hybrid Retriever]      → BM25 (keyword) + FAISS (semantic) merged via RRF
        │
        ▼
[Groq Generator]      → grounded reply + JSON recommendation block
        │
        ▼
[Validation Layer]      → every URL cross-checked against catalog; no hallucinations
        │
        ▼
ChatResponse
```

See [`docs/approach_document.md`](docs/approach_document.md) for the full technical write-up.

---

## License

This project was developed exclusively as a technical submission for the **SHL AI Intern Assignment**.
