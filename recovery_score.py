"""
recovery_score.py
------------------
Deterministic recovery score calculator. No LLM involved here on purpose —
this is the "tool use" / structured-data part of the pipeline, and the score
must be reproducible and explainable.

Input is a single flat dict representing "today's" metrics plus their
baseline columns. Every row of p01_sanitized.csv is already shaped this way,
so in CSV mode you can pass a row straight in. In manual-override mode,
start from the latest CSV row and overwrite only the fields the user typed
over (sleep_hours, resting_hr, hrv_ms, stress_score, training_load) —
baseline_* fields stay as pulled from history, since a single hypothetical
"today" has no trend of its own to compute a baseline from.

Expected keys (all numeric, all required):
    sleep_hours, resting_hr, hrv_ms, stress_score, training_load, restlessness
    sleep_hours_7d_baseline, resting_hr_7d_baseline,
    hrv_ms_7d_baseline, restlessness_7d_baseline
"""

from dataclasses import dataclass, field


@dataclass
class RecoveryResult:
    score: int
    level: str                 # "High" | "Medium" | "Low"
    reason_codes: list = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)  # rule -> points deducted


# Thresholds calibrated against p01_sanitized.csv's actual value distributions
# (checked via percentile analysis, not guessed) so that each rule fires on a
# realistic minority of days instead of never (or almost always) triggering.
# These are a teaching-example formula per the project brief — not a
# validated clinical/physiological model. If you swap in a different user's
# CSV, re-check percentiles before trusting these numbers.
THRESHOLDS = {
    "sleep_below_baseline_severe_hrs": 0.8,   # ~p10: notably below baseline
    "sleep_below_baseline_mild_hrs": 0.45,    # ~p25: mildly below baseline
    "resting_hr_above_baseline_pct": 0.015,   # ~p75-p90: this dataset's HR barely
                                               # moves (max ~5% dev), so 1.5% is
                                               # already a meaningfully elevated day
    "hrv_below_baseline_pct": 0.08,           # ~p10-p15: below-baseline HRV dip
    "stress_score_high": 3,                   # this user's observed ceiling (2-3
                                               # range) — see note in docstring
    "training_load_high": 4.0,                # ~p75: a harder-than-usual session
    "restlessness_above_baseline_pct": 0.25,  # ~p80: notably restless night
}

DEDUCTIONS = {
    "sleep_below_baseline_severe": 15,
    "sleep_below_baseline_mild": 10,
    "resting_hr_above_baseline": 15,
    "hrv_below_baseline": 10,
    "stress_score_high": 5,   # lowered from doc's example 10: this dataset's
                               # stress_score has almost no range (2-3), so a
                               # heavier weight would flatten every day's score
    "training_load_high": 10,
    "restlessness_above_baseline": 5,
}


def calculate_recovery_score(metrics: dict) -> RecoveryResult:
    score = 100
    reasons = []
    breakdown = {}

    def deduct(rule: str, points: int, reason_code: str):
        nonlocal score
        score -= points
        reasons.append(reason_code)
        breakdown[rule] = points

    # --- Sleep duration vs baseline ---
    sleep_gap = metrics["sleep_hours_7d_baseline"] - metrics["sleep_hours"]
    if sleep_gap > THRESHOLDS["sleep_below_baseline_severe_hrs"]:
        deduct("sleep_below_baseline_severe", DEDUCTIONS["sleep_below_baseline_severe"],
               "sleep_duration_well_below_baseline")
    elif sleep_gap > THRESHOLDS["sleep_below_baseline_mild_hrs"]:
        deduct("sleep_below_baseline_mild", DEDUCTIONS["sleep_below_baseline_mild"],
               "sleep_duration_below_baseline")

    # --- Resting HR vs baseline ---
    hr_baseline = metrics["resting_hr_7d_baseline"]
    if hr_baseline > 0:
        hr_pct_above = (metrics["resting_hr"] - hr_baseline) / hr_baseline
        if hr_pct_above > THRESHOLDS["resting_hr_above_baseline_pct"]:
            deduct("resting_hr_above_baseline", DEDUCTIONS["resting_hr_above_baseline"],
                   "resting_hr_above_baseline")

    # --- HRV vs baseline ---
    hrv_baseline = metrics["hrv_ms_7d_baseline"]
    if hrv_baseline > 0:
        hrv_pct_below = (hrv_baseline - metrics["hrv_ms"]) / hrv_baseline
        if hrv_pct_below > THRESHOLDS["hrv_below_baseline_pct"]:
            deduct("hrv_below_baseline", DEDUCTIONS["hrv_below_baseline"],
                   "hrv_below_baseline")

    # --- Stress score (absolute, not baseline-relative) ---
    if metrics["stress_score"] >= THRESHOLDS["stress_score_high"]:
        deduct("stress_score_high", DEDUCTIONS["stress_score_high"],
               "stress_score_elevated")

    # --- Training load yesterday (absolute) ---
    if metrics["training_load"] >= THRESHOLDS["training_load_high"]:
        deduct("training_load_high", DEDUCTIONS["training_load_high"],
               "high_training_load_yesterday")

    # --- Restlessness vs baseline (stand-in for "awake time during sleep") ---
    restlessness_baseline = metrics["restlessness_7d_baseline"]
    if restlessness_baseline > 0:
        restlessness_pct_above = (
            (metrics["restlessness"] - restlessness_baseline) / restlessness_baseline
        )
        if restlessness_pct_above > THRESHOLDS["restlessness_above_baseline_pct"]:
            deduct("restlessness_above_baseline", DEDUCTIONS["restlessness_above_baseline"],
                   "restlessness_above_baseline")

    score = max(score, 0)

    if score >= 80:
        level = "High"
    elif score >= 60:
        level = "Medium"
    else:
        level = "Low"

    return RecoveryResult(score=score, level=level, reason_codes=reasons, breakdown=breakdown)


if __name__ == "__main__":
    # Quick manual smoke test
    example = {
        "sleep_hours": 5.0,
        "sleep_hours_7d_baseline": 6.0,
        "resting_hr": 54.3,
        "resting_hr_7d_baseline": 52.5,
        "hrv_ms": 44.3,
        "hrv_ms_7d_baseline": 55.0,
        "stress_score": 3,
        "training_load": 4.5,
        "restlessness": 0.095,
        "restlessness_7d_baseline": 0.073,
    }
    result = calculate_recovery_score(example)
    print(f"Score: {result.score}/100  Level: {result.level}")
    print("Reasons:", result.reason_codes)
    print("Breakdown:", result.breakdown)
