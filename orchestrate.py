"""
orchestrate.py
---------------
The agentic orchestration layer. This is the piece that turns four separate
tools (safety_checker, wearable data reader, recovery_score, retrieve) into
one coherent wellness recommendation.

"Agentic" here means: the path taken through the pipeline is decided by the
data at runtime, not hardcoded per query. A safety flag short-circuits
everything else. The recovery level (High/Medium/Low) changes which RAG
query is issued and which workout guidance gets retrieved. This matches the
project brief's definition of the agentic requirement (conditional routing
based on tool outputs) -- it does not require a multi-agent framework.

Pipeline:
    user_text
        -> check_safety()                              [safety_checker.py]
        -> (if blocked) return safety response, STOP
        -> load today's wearable row + baseline          [this file]
        -> apply manual overrides if provided            [this file]
        -> calculate_recovery_score()                    [recovery_score.py]
        -> build a RAG query based on recovery level      [this file]
        -> retrieve_guidance()                            [retrieve.py]
        -> generate final answer (Gemini, falls back to template)

Usage:
    from orchestrate import run_pipeline

    result = run_pipeline(
        user_text="Should I do leg day today?",
        csv_path="data/raw/p01_sanitized.csv",
    )
    print(result["final_response"])
"""

import os
import csv
from pathlib import Path

from safety_checker import check_safety
from recovery_score import calculate_recovery_score
from retrieve import retrieve_guidance

# Fields pulled straight from a CSV row into the recovery score calculator.
# These are the only fields recovery_score.py actually needs.
METRIC_FIELDS = [
    "sleep_hours", "sleep_hours_7d_baseline",
    "resting_hr", "resting_hr_7d_baseline",
    "hrv_ms", "hrv_ms_7d_baseline",
    "stress_score", "training_load",
    "restlessness", "restlessness_7d_baseline",
]

# Only these fields can be manually overridden in hybrid UI mode.
# Baseline_* fields are never user-editable -- a single hypothetical "today"
# has no trend of its own, so baselines always come from real history.
OVERRIDABLE_FIELDS = [
    "sleep_hours", "resting_hr", "hrv_ms", "stress_score", "training_load", "restlessness",
]


# ---------------------------------------------------------------------
# 1. Wearable data loading
# ---------------------------------------------------------------------
def load_wearable_row(csv_path: str, target_date: str | None = None) -> dict:
    """
    Load a single day's wearable row. Defaults to the most recent date in
    the file if target_date is not given. Returns a dict with the metric
    fields already converted to float.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    if target_date:
        matches = [r for r in rows if r["date"] == target_date]
        if not matches:
            raise ValueError(f"No wearable row found for date {target_date}")
        row = matches[0]
    else:
        row = sorted(rows, key=lambda r: r["date"])[-1]

    metrics = {field: float(row[field]) for field in METRIC_FIELDS}
    metrics["date"] = row["date"]
    return metrics


def apply_overrides(metrics: dict, overrides: dict | None) -> dict:
    """
    Return a new metrics dict with user-supplied overrides applied.
    Silently ignores any override key that isn't in OVERRIDABLE_FIELDS,
    so a manual-input form can't accidentally corrupt baseline values.
    """
    result = dict(metrics)
    if overrides:
        for key, value in overrides.items():
            if key in OVERRIDABLE_FIELDS and value is not None:
                result[key] = float(value)
    return result


# ---------------------------------------------------------------------
# 2. Conditional routing: recovery level -> RAG query
# ---------------------------------------------------------------------
RAG_QUERIES_BY_LEVEL = {
    "Low": "active recovery routines, low intensity workout, hydration, sleep recovery guidance",
    "Medium": "reduced intensity workout guidance, moderate training volume, warm-up and cool-down",
    "High": "normal workout progression, strength or cardio training guidance",
}


def build_rag_query(recovery_level: str) -> str:
    return RAG_QUERIES_BY_LEVEL.get(recovery_level, RAG_QUERIES_BY_LEVEL["Medium"])


# ---------------------------------------------------------------------
# 3. Final answer generation -- Gemini, with a deterministic fallback
# ---------------------------------------------------------------------
SYSTEM_INSTRUCTIONS = """You are a wellness and workout-readiness assistant. You are NOT a doctor.

Rules you must always follow:
- Never diagnose a medical condition or claim the user has one.
- Never prescribe medication.
- Never claim wearable-derived sleep stages or HRV are clinically exact -- call them estimates.
- Base your answer only on the structured metrics and retrieved guidance provided below. Do not invent numbers or guidance not given to you.
- Use the retrieved guidance to inform your wording, but do not quote it verbatim, and do not mention document names, filenames, or sources -- write it as your own advice, not a citation.
- Keep the whole answer concise -- a few sentences per section, not paragraphs.
- Always end with a short safety disclaimer noting this is wellness guidance, not medical advice, and to consult a professional for persistent or serious symptoms.

Format your answer with these sections, in this order, and nothing else:
Recovery level:
Key signals:
Suggested workout:
What to avoid:
Why:
Safety note:
"""


def _build_llm_prompt(user_text, metrics, recovery_result, retrieved_chunks, pregnancy_note):
    signals = "\n".join(f"- {code.replace('_', ' ')}" for code in recovery_result.reason_codes) or "- No notable deviations from baseline"
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in retrieved_chunks) or "No guidance retrieved."

    extra = f"\n\nNote: the user's message involves pregnancy -- keep guidance general and defer exercise specifics to their doctor.\n{pregnancy_note}" if pregnancy_note else ""

    return f"""{SYSTEM_INSTRUCTIONS}

User question: "{user_text}"

Today's date: {metrics.get('date', 'unknown')}
Recovery score: {recovery_result.score}/100 ({recovery_result.level} recovery)
Reason codes:
{signals}

Retrieved wellness guidance:
{context}
{extra}
"""


def generate_llm_response(user_text, metrics, recovery_result, retrieved_chunks, pregnancy_note=None) -> str | None:
    """
    Calls Gemini. Returns None (rather than raising) on any failure --
    missing API key, network issue, quota error, etc. -- so the caller can
    fall back to the template response and the app never hard-crashes on
    an LLM problem.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
        model = genai.GenerativeModel(model_name)

        prompt = _build_llm_prompt(user_text, metrics, recovery_result, retrieved_chunks, pregnancy_note)
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        return text if text else None
    except Exception as e:
        print(f"[orchestrate] Gemini call failed, falling back to template: {e}")
        return None


def generate_template_response(user_text, metrics, recovery_result, retrieved_chunks, pregnancy_note=None) -> str:
    """Deterministic fallback -- no LLM required. Used if no API key is set
    or the Gemini call fails for any reason.

    Note: retrieved_chunks informs which RAG query was issued (see
    build_rag_query) and is still returned separately in run_pipeline()'s
    result dict for logging/traceability, but is intentionally NOT quoted
    verbatim in the user-facing response -- dumping raw document snippets
    reads like a database dump, not a wellness answer, and section 18 of
    the project brief's own output format doesn't include one either."""
    signals = "\n".join(f"- {code.replace('_', ' ')}" for code in recovery_result.reason_codes) or "- No notable deviations from your recent baseline"

    workout_by_level = {
        "Low": "20-30 minutes easy walking, 10 minutes mobility work, light stretching. Avoid high-intensity intervals today.",
        "Medium": "Your planned session is fine, but reduce volume by 20-30% and avoid training to failure.",
        "High": "You can follow your planned workout. Start with a proper warm-up and monitor how you feel.",
    }
    avoid_by_level = {
        "Low": "Max-effort lifting, sprints, training to failure, long endurance sessions.",
        "Medium": "Training to failure and skipping warm-up/cool-down.",
        "High": "Skipping your warm-up even on a good day.",
    }

    pregnancy_block = f"\n\nNote: {pregnancy_note}" if pregnancy_note else ""

    return f"""Recovery level: {recovery_result.level} ({recovery_result.score}/100)

Key signals:
{signals}

Suggested workout:
{workout_by_level[recovery_result.level]}

What to avoid:
{avoid_by_level[recovery_result.level]}

Why:
Based on your wearable-style data compared to your recent baseline, today's recovery signals suggest this level of intensity is appropriate.
{pregnancy_block}

Safety note:
This is wellness guidance based on wearable-style estimates, not medical advice. Wearable metrics are estimates, not clinical measurements. If fatigue persists or you notice symptoms such as chest pain, dizziness, fainting, or unusual breathlessness, please consult a qualified medical professional."""


# ---------------------------------------------------------------------
# 4. Main orchestration entry point
# ---------------------------------------------------------------------
def run_pipeline(
    user_text: str,
    csv_path: str,
    target_date: str | None = None,
    overrides: dict | None = None,
    use_llm: bool = True,
) -> dict:
    """
    Runs the full agentic pipeline and returns a dict containing every
    intermediate result -- this same dict is what you'd write straight to
    logs.csv for the monitoring requirement.
    """
    # Step 1: Safety check, always first.
    safety_result = check_safety(user_text)
    if safety_result.blocks_workout_advice:
        return {
            "user_text": user_text,
            "safety_flagged": True,
            "safety_category": safety_result.category,
            "final_response": safety_result.response_message,
            "used_llm": False,
            "recovery_score": None,
            "recovery_level": None,
            "rag_query": None,
            "retrieved_chunks": [],
            "metrics_used": None,
        }

    pregnancy_note = safety_result.response_message if safety_result.category == "pregnancy_caution" else None

    # Step 2: Wearable data + baseline.
    raw_metrics = load_wearable_row(csv_path, target_date)
    metrics = apply_overrides(raw_metrics, overrides)

    # Step 3: Recovery score (deterministic tool, no LLM).
    recovery_result = calculate_recovery_score({k: metrics[k] for k in METRIC_FIELDS})

    # Step 4: Conditional routing -> RAG query -> retrieval.
    rag_query = build_rag_query(recovery_result.level)
    retrieved_chunks = retrieve_guidance(rag_query, k=3)

    # Step 5: Final answer -- try Gemini, fall back to template.
    final_response = None
    used_llm = False
    if use_llm:
        final_response = generate_llm_response(user_text, metrics, recovery_result, retrieved_chunks, pregnancy_note)
        used_llm = final_response is not None

    if final_response is None:
        final_response = generate_template_response(user_text, metrics, recovery_result, retrieved_chunks, pregnancy_note)

    return {
        "user_text": user_text,
        "safety_flagged": safety_result.flagged,
        "safety_category": safety_result.category,
        "metrics_used": metrics,
        "recovery_score": recovery_result.score,
        "recovery_level": recovery_result.level,
        "reason_codes": recovery_result.reason_codes,
        "rag_query": rag_query,
        "retrieved_chunks": retrieved_chunks,
        "final_response": final_response,
        "used_llm": used_llm,
    }


if __name__ == "__main__":
    csv_path = str(Path(__file__).parent / "data" / "raw" / "p01_sanitized.csv")

    print("=== Normal query (no overrides, latest date) ===")
    result = run_pipeline("Should I do leg day today?", csv_path, use_llm=False)
    print(result["final_response"])

    print("\n=== Safety escalation ===")
    result = run_pipeline("I have chest tightness and dizziness. Should I still work out?", csv_path, use_llm=False)
    print(result["final_response"])