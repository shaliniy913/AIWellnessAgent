"""
streamlit_app.py
------------------
Single entrypoint for the whole app. Anyone cloning this repo should only
ever need to run:

    pip install -r requirements.txt
    streamlit run streamlit_app.py

No separate ingestion step, no manual Chroma build. On first launch this
file checks whether the vector index exists and builds it automatically
from guidance_docs/*.pdf if it doesn't (see _ensure_index_built below).
"""

import csv
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads .env for GEMINI_API_KEY / GOOGLE_API_KEY if present

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "data" / "raw" / "p01_sanitized.csv"
CHROMA_PATH = BASE_DIR / "chroma_db"
LOGS_PATH = BASE_DIR / "logs.csv"

st.set_page_config(page_title="AI Wellness & Recovery Agent", layout="wide")


# ---------------------------------------------------------------------
# Self-healing setup: build the vector index on first run if missing
# ---------------------------------------------------------------------
def _ensure_index_built():
    needs_build = not CHROMA_PATH.exists() or not any(CHROMA_PATH.iterdir())
    if needs_build:
        with st.spinner("First-time setup: indexing wellness guidance documents (only happens once)..."):
            try:
                import ingest
                ingest.main()
            except FileNotFoundError as e:
                st.error(
                    f"Couldn't build the guidance index: {e}\n\n"
                    "Make sure guidance_docs/ contains at least one PDF, then reload this page."
                )
                st.stop()


_ensure_index_built()

# Imported after the index-build check so a missing chromadb collection
# doesn't crash the app before _ensure_index_built has a chance to fix it.
from orchestrate import run_pipeline, OVERRIDABLE_FIELDS  # noqa: E402


# ---------------------------------------------------------------------
# Data loading (cached so the CSV isn't re-read on every interaction)
# ---------------------------------------------------------------------
@st.cache_data
def load_dataframe():
    if not CSV_PATH.exists():
        st.error(f"Wearable CSV not found at {CSV_PATH}. Place your data there and reload.")
        st.stop()
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    return df.sort_values("date")


def log_interaction(row: dict):
    """Append one interaction to logs.csv, writing the header if the file is new."""
    file_exists = LOGS_PATH.exists()
    with open(LOGS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def looks_like_plan_change_request(text: str) -> bool:
    """Heuristic to decide whether a query needs human approval before being
    treated as applied (per the doc's Step 9 / weekly-plan-change flow).
    Uses an action-word + plan-word combination rather than fixed phrases,
    since "update my plan" and "update my workout plan for this week" (the
    doc's own Query 4 example) don't share a common substring."""
    text_lower = text.lower()
    action_words = ["update", "change", "modify", "adjust", "revise"]
    plan_words = ["plan", "schedule", "routine"]
    has_action = any(w in text_lower for w in action_words)
    has_plan = any(w in text_lower for w in plan_words)
    return has_action and has_plan


df = load_dataframe()
all_dates = df["date"].dt.strftime("%Y-%m-%d").tolist()

# ---------------------------------------------------------------------
# Sidebar: date selection + manual override
# ---------------------------------------------------------------------
st.sidebar.title("Today's Data")

selected_date = st.sidebar.selectbox("Simulate 'today' as:", options=all_dates, index=len(all_dates) - 1)
today_row = df[df["date"].dt.strftime("%Y-%m-%d") == selected_date].iloc[0]

st.sidebar.markdown("---")
use_override = st.sidebar.checkbox("Override today's values (what-if mode)")

overrides = None
if use_override:
    st.sidebar.caption("Baseline values stay fixed — only today's readings change.")
    overrides = {
        "sleep_hours": st.sidebar.slider("Sleep hours", 0.0, 12.0, float(today_row["sleep_hours"]), 0.25),
        "resting_hr": st.sidebar.slider("Resting HR (bpm)", 40.0, 100.0, float(today_row["resting_hr"]), 0.5),
        "hrv_ms": st.sidebar.slider("HRV (ms)", 10.0, 100.0, float(today_row["hrv_ms"]), 0.5),
        "stress_score": st.sidebar.slider("Stress score", 1, 5, int(today_row["stress_score"])),
        "training_load": st.sidebar.slider("Training load yesterday", 0.0, 10.0, float(today_row["training_load"]), 0.1),
    }

use_llm = st.sidebar.checkbox("Use Gemini for the final answer", value=True)
if use_llm and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
    st.sidebar.warning("No GEMINI_API_KEY found in .env — answers will fall back to the built-in template.")


# ---------------------------------------------------------------------
# Main: dashboard
# ---------------------------------------------------------------------
st.title("🏃 AI Wellness, Recovery & Workout Readiness Agent")
st.caption("A wellness assistant, not a medical service. See the safety note on every answer.")

st.subheader(f"Snapshot for {selected_date}")
cols = st.columns(5)
cols[0].metric("Sleep (hrs)", f"{today_row['sleep_hours']:.1f}", f"{today_row['sleep_hours'] - today_row['sleep_hours_7d_baseline']:+.1f} vs 7d avg")
cols[1].metric("Resting HR", f"{today_row['resting_hr']:.0f}", f"{today_row['resting_hr'] - today_row['resting_hr_7d_baseline']:+.1f} vs 7d avg")
cols[2].metric("HRV (ms)", f"{today_row['hrv_ms']:.0f}", f"{today_row['hrv_ms'] - today_row['hrv_ms_7d_baseline']:+.1f} vs 7d avg")
cols[3].metric("Stress score", f"{today_row['stress_score']:.0f}")
cols[4].metric("Training load (yest.)", f"{today_row['training_load']:.1f}")

with st.expander("Show recent trend (last 30 days)"):
    recent = df.tail(30).set_index("date")
    st.line_chart(recent[["sleep_hours", "resting_hr", "hrv_ms"]])

st.markdown("---")


# ---------------------------------------------------------------------
# Chat-style Q&A
# ---------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

st.subheader("Ask about today's readiness")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_text = st.chat_input("e.g. Should I do leg day today?")

if user_text:
    st.session_state.history.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    result = run_pipeline(
        user_text=user_text,
        csv_path=str(CSV_PATH),
        target_date=selected_date,
        overrides=overrides,
        use_llm=use_llm,
    )

    with st.chat_message("assistant"):
        st.markdown(result["final_response"])

        if not result["safety_flagged"] or result["safety_category"] == "pregnancy_caution":
            with st.expander("Show details"):
                st.write(f"**Recovery score:** {result['recovery_score']}/100 — {result['recovery_level']}")
                reason_codes = result.get("reason_codes", [])
                if reason_codes:
                    readable = ", ".join(code.replace("_", " ") for code in reason_codes)
                    st.write(f"**Key factors:** {readable}")
                else:
                    st.write("**Key factors:** no notable deviations from baseline")
                st.caption(f"Answer generated by {'Gemini' if result['used_llm'] else 'built-in template'}")

    st.session_state.history.append({"role": "assistant", "content": result["final_response"]})

    approval_status = "not_applicable"

    # Human approval flow for plan-change requests (doc Step 9)
    if looks_like_plan_change_request(user_text) and not result["safety_flagged"]:
        st.session_state.pending_approval = result
        approval_status = "pending"

    log_interaction({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "date_simulated": selected_date,
        "user_query": user_text,
        "safety_flagged": result["safety_flagged"],
        "safety_category": result["safety_category"],
        "recovery_score": result["recovery_score"],
        "recovery_level": result["recovery_level"],
        "rag_query": result.get("rag_query"),
        "used_llm": result["used_llm"],
        "override_applied": bool(overrides),
        "approval_status": approval_status,
    })

# ---------------------------------------------------------------------
# Human approval widget (only shown for plan-change requests)
# ---------------------------------------------------------------------
if st.session_state.pending_approval:
    st.markdown("---")
    st.warning("This request would change your weekly plan. Please confirm before it's applied.")
    c1, c2, c3 = st.columns(3)
    decision = None
    if c1.button("✅ Approve"):
        decision = "approved"
    if c2.button("✏️ Modify"):
        decision = "modify_requested"
    if c3.button("❌ Reject"):
        decision = "rejected"

    if decision:
        st.session_state.pending_approval = None
        log_interaction({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "date_simulated": selected_date,
            "user_query": "[approval_decision]",
            "safety_flagged": False,
            "safety_category": None,
            "recovery_score": None,
            "recovery_level": None,
            "rag_query": None,
            "used_llm": None,
            "override_applied": None,
            "approval_status": decision,
        })
        st.success(f"Decision recorded: {decision}")
        st.rerun()