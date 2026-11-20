**An LLM-powered resume analyzer that scores your resume against any job description, explains the gaps, rewrites your bullets and preps you for the interview.**

Built with Python, LLMs (via any OpenAI-compatible API such as Groq), local embeddings, and Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red) ![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20OpenAI%20%7C%20Ollama-green) ![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)

<!-- Add a screenshot or GIF of the app here: ![demo](docs/demo.png) -->

---

## ✨ Features

| Feature | What it does | GenAI technique |
|---|---|---|
| 📊 **Fit report** | Overall score, skill-by-skill match with evidence, strengths, gaps, action plan | Structured output (JSON + Pydantic), embeddings |
| 🔎 **Semantic skill matching** | Finds skills even when worded differently ("NLP" vs "natural language processing") | Sentence embeddings + cosine similarity |
| ✍️ **Resume tailoring** | Rewrites your bullets for the job without inventing experience | Prompt engineering with guardrails |
| 🎤 **Interview prep** | Role-specific technical, project, behavioral and HR questions with answer tips | Context-grounded generation |
| 💬 **Chat with your resume** | Ask questions; answers come only from your resume | RAG (chunking → embeddings → retrieval → generation) |
| ⬇️ **Export** | Download the full report as Markdown | — |

## 🏗️ Architecture

```
            ┌──────────────┐         ┌─────────────────────┐
 Resume ───►│ Parser       │──text──►│ Chunker             │──chunks──┐
 (PDF/DOCX) │ pypdf/docx   │         │ paragraph-aware +   │          ▼
            └──────────────┘         │ overlap             │   ┌──────────────┐
                                     └─────────────────────┘   │ Vector store │
                                                               │ (numpy +     │
 Job desc ──► LLM: extract requirements (JSON schema) ──┐      │ fastembed)   │
                                                        ▼      └──────┬───────┘
                                      Embedding skill matcher ◄───────┤
                                      (exact match + cosine)          │ top-k
                                                        │             ▼
                                                        └──► LLM: grounded fit assessment
                                                                      │
                     Score = 50% skill coverage + 50% LLM judgement ◄─┘
```

**Why a hybrid score?** Pure LLM scores are inconsistent between runs. The embedding-based skill coverage is deterministic and explainable; the LLM adds judgement about seniority, depth and project quality. Blending both gives a score that is stable *and* sensible.

**Reliable JSON from LLMs:** every structured call is validated with a Pydantic schema. If the model returns invalid JSON, the validation error is fed back and the model retries (see `jobfit/llm.py`).

## Output Dashboard

<img width="1379" height="834" alt="Screenshot 2026-11-20 at 1 50 06 PM" src="https://github.com/user-attachments/assets/26524003-b712-4984-95e5-9d7215e9931c" />

## 🚀 Quick start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/jobfit-ai.git
cd jobfit-ai
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get a free API key

1. Sign up at [console.groq.com](https://console.groq.com) (free tier, very fast inference).
2. Create an API key.
3. Copy `.env.example` to `.env` and paste your key:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

> Model names change over time. If you get a "model not found" error, pick a current model from [Groq's models page](https://console.groq.com/docs/models) and update `LLM_MODEL` in `.env`.

### 3. Run

```bash
streamlit run app.py
```

Open http://localhost:8501, tick **Use sample resume** and click **Analyze fit** to try it instantly.

**Command-line version:**

```bash
python cli.py --resume samples/resume.txt --jd samples/job_description.txt --out report.md
```

### Use a different LLM provider

Any OpenAI-compatible API works. Just change `.env`:

| Provider | `LLM_BASE_URL` | Notes |
|---|---|---|
| Groq (default) | `https://api.groq.com/openai/v1` | Free tier |
| OpenAI | `https://api.openai.com/v1` | e.g. `gpt-4o-mini` |
| Ollama (local, free, offline) | `http://localhost:11434/v1` | `LLM_API_KEY=ollama`, e.g. `llama3.1` |

## 🧪 Tests

The test suite uses a fake LLM and a fake embedder, so it runs offline with no API key:

```bash
pytest -q
```

Tests cover chunking, file parsing, JSON extraction and retry logic, vector search, skill matching (including whole-word matching so "Java" ≠ "JavaScript"), the scoring formula, RAG context injection, and the report generator. CI runs them on every push via GitHub Actions.

## 📁 Project structure

```
jobfit-ai/
├── app.py                 # Streamlit web app
├── cli.py                 # Command-line interface
├── jobfit/
│   ├── analyzer.py        # Core pipeline: extract → match → assess → RAG chat
│   ├── llm.py             # OpenAI-compatible client, retries, JSON validation
│   ├── embeddings.py      # fastembed wrapper + in-memory vector store
│   ├── chunking.py        # Paragraph/sentence-aware chunking with overlap
│   ├── parsing.py         # PDF / DOCX / TXT text extraction
│   ├── prompts.py         # All prompt templates
│   ├── schemas.py         # Pydantic schemas for structured LLM output
│   ├── report.py          # Markdown report export
│   └── config.py          # Settings from .env
├── samples/               # Sample resume and job description
├── tests/                 # pytest suite (offline)
└── .github/workflows/     # CI
```
## 🛣️ Roadmap

- [ ] Compare one resume against multiple job descriptions at once
- [ ] FastAPI backend + Docker image
- [ ] LLM-as-judge evaluation set to measure answer quality and hallucinations
- [ ] Persist analyses in SQLite

## ⚠️ Disclaimer

Scores are guidance, not a hiring decision. The app is prompted never to invent experience, but always review AI-generated resume text before using it.

## 📄 License

MIT
