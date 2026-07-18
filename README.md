# AIWellness — AI Wellness, Recovery & Workout Readiness Agent

A wellness assistant that reads wearable-style data, compares it to your
recent baseline, calculates a recovery score, retrieves relevant guidance
from a small knowledge base, and gives a safe, explainable workout
suggestion. It is **not** a medical diagnosis tool — see the safety note
included in every answer.

This is not a chatbot that generates generic workout advice. It reads your
actual wearable metrics, compares them to your own baseline, retrieves
relevant recovery guidance, and routes you to a High/Medium/Low recovery
path — with a human approval step before any weekly plan change is applied.

## How it works

```
user question
    -> safety check                  (safety_checker.py)
    -> (if unsafe) STOP, redirect to a medical professional
    -> read today's wearable row + baseline    (orchestrate.py)
    -> calculate recovery score                (recovery_score.py)
    -> route to a RAG query based on recovery level
    -> retrieve relevant guidance               (rag_pipeline/retrieve.py)
    -> generate final answer (Gemini, falls back to a template)
```

The routing step is what makes this "agentic" per the project brief: the
path taken depends on the data, not a fixed script.

## Project structure

```
AIWellnessAgent/
    1. streamlit_app.py         <- run this — single entrypoint for the whole app
    2. orchestrate.py             <- the agentic pipeline (see diagram above)
    3. recovery_score.py           <- deterministic recovery score calculator
    4. safety_checker.py            <- keyword/pattern-based safety guardrail
    5. data/
         raw/
           p01_sanitized.csv     <- wearable-style metrics + baselines
    6. guidance_docs/               <- source PDFs (sleep, recovery, safety, etc.)
    7. chroma_db/                    <- generated on first run, not committed
    8. ingest.py                      <- builds the vector index (auto-run if missing)
    9. retrieve.py                     <- retrieve_guidance(), called by orchestrate.py
    10. README.md                       <- RAG-specific setup/troubleshooting notes
    11. tests/
          test_recovery_score.py
          test_safety_checker.py
    12. requirements.txt                <- one file, covers the whole project
    13. .env.example
    14. .gitignore
    15. logs.csv                          <- generated per-interaction monitoring log
```

## Setup

```bash
git clone <this-repo-url>
cd AIWellnessAgent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env            # creates YOUR OWN local .env from the blank
                                 # template -- then open .env and paste in
                                 # your own Gemini key. .env is git-ignored,
                                 # so this never gets committed or shared.
```

Get a free Gemini API key at https://aistudio.google.com/apikey.

## Run

```bash
streamlit run streamlit_app.py
```

One command. On first launch, the app checks whether `/chroma_db/`
exists and automatically builds it from `/guidance_docs/*.pdf` if
it doesn't (takes ~10-20 seconds, only happens once — later launches are
instant). No API key is needed for this step; embeddings are generated
locally with `sentence-transformers` (`all-MiniLM-L6-v2`).

If you don't set a Gemini key, the app still runs fine — it falls back to a
deterministic template-based answer instead of an LLM-generated one. There's
also a sidebar toggle to force template mode even when a key is present,
useful for testing the deterministic parts of the pipeline in isolation.

## Using the app

- **Sidebar date picker** — simulates "today" using any date from the CSV
- **"Override today's values"** — what-if mode: drag sliders to hypothetical
  sleep/HR/HRV/stress/training-load values while baselines stay fixed to
  real history
- **Chat box** — ask things like *"Should I do leg day today?"* or *"My
  sleep was 7 hours but I feel exhausted, why?"*
- **Show details** — expandable per-answer summary of the recovery score
  and key contributing factors
- **Approve / Modify / Reject** — appears automatically when a question
  implies changing your weekly plan (per the project's human-approval
  requirement)

## Running tests

```bash
pytest tests/ -v
```

Covers recovery score threshold/stacking/boundary logic and the safety
checker's three categories (medical emergency, disordered eating, pregnancy
caution), including false-positive checks on ordinary fitness phrasing.

## Monitoring

Every interaction is appended to `logs.csv` at the project root: timestamp,
query, safety flag, recovery score/level, RAG query used, which engine
answered (Gemini or template), and any approval decision. This is the
traceability record for the project's monitoring requirement — kept out of
the chat UI itself to keep answers readable, but fully preserved here.

## Notes on safety

This system does not diagnose medical conditions, prescribe medication, or
give clinical advice. It explicitly refuses to give workout guidance when a
user mentions warning symptoms (chest pain, dizziness, fainting, etc.) or
requests an extreme/rapid weight-loss plan, and instead points to a
qualified medical professional. See `safety_checker.py` for the full rule
set and `README.md` for RAG-specific setup and troubleshooting.
