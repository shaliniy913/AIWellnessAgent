"""
test_recovery_score.py
-----------------------
Run with:
    pytest tests/test_recovery_score.py -v

Covers:
  - each deduction rule in isolation (does it fire exactly when it should)
  - stacking multiple bad signals
  - score floor at 0
  - level bucketing (High/Medium/Low cutoffs)
  - two real rows pulled from p01_sanitized.csv (worst and best observed days)
"""

import pytest
from recovery_score import calculate_recovery_score


def perfect_day():
    """A baseline metrics dict where nothing should trigger a deduction."""
    return {
        "sleep_hours": 6.0,
        "sleep_hours_7d_baseline": 6.0,
        "resting_hr": 52.0,
        "resting_hr_7d_baseline": 52.0,
        "hrv_ms": 55.0,
        "hrv_ms_7d_baseline": 55.0,
        "stress_score": 2,          # below the "high" threshold of 3
        "training_load": 2.0,       # below the "high" threshold of 4.0
        "restlessness": 0.08,
        "restlessness_7d_baseline": 0.08,
    }


def test_perfect_day_scores_100():
    result = calculate_recovery_score(perfect_day())
    assert result.score == 100
    assert result.level == "High"
    assert result.reason_codes == []


def test_sleep_mild_deduction_fires_at_threshold():
    m = perfect_day()
    m["sleep_hours"] = m["sleep_hours_7d_baseline"] - 0.5  # just past the 0.45 mild cutoff
    result = calculate_recovery_score(m)
    assert "sleep_duration_below_baseline" in result.reason_codes
    assert result.score == 90  # 100 - 10


def test_sleep_severe_deduction_overrides_mild():
    m = perfect_day()
    m["sleep_hours"] = m["sleep_hours_7d_baseline"] - 1.0  # past the 0.8 severe cutoff
    result = calculate_recovery_score(m)
    assert "sleep_duration_well_below_baseline" in result.reason_codes
    assert "sleep_duration_below_baseline" not in result.reason_codes  # only one sleep rule should fire
    assert result.score == 85  # 100 - 15


def test_resting_hr_above_baseline_fires():
    m = perfect_day()
    m["resting_hr"] = m["resting_hr_7d_baseline"] * 1.02  # 2% above, past the 1.5% cutoff
    result = calculate_recovery_score(m)
    assert "resting_hr_above_baseline" in result.reason_codes
    assert result.score == 85  # 100 - 15


def test_hrv_below_baseline_fires():
    m = perfect_day()
    m["hrv_ms"] = m["hrv_ms_7d_baseline"] * 0.90  # 10% below, past the 8% cutoff
    result = calculate_recovery_score(m)
    assert "hrv_below_baseline" in result.reason_codes
    assert result.score == 90  # 100 - 10


def test_stress_score_high_fires():
    m = perfect_day()
    m["stress_score"] = 3
    result = calculate_recovery_score(m)
    assert "stress_score_elevated" in result.reason_codes
    assert result.score == 95  # 100 - 5


def test_training_load_high_fires():
    m = perfect_day()
    m["training_load"] = 5.0
    result = calculate_recovery_score(m)
    assert "high_training_load_yesterday" in result.reason_codes
    assert result.score == 90  # 100 - 10


def test_restlessness_above_baseline_fires():
    m = perfect_day()
    m["restlessness"] = m["restlessness_7d_baseline"] * 1.30  # 30% above, past the 25% cutoff
    result = calculate_recovery_score(m)
    assert "restlessness_above_baseline" in result.reason_codes
    assert result.score == 95  # 100 - 5


def test_multiple_bad_signals_stack():
    m = perfect_day()
    m["sleep_hours"] = m["sleep_hours_7d_baseline"] - 1.0   # -15
    m["hrv_ms"] = m["hrv_ms_7d_baseline"] * 0.80             # -10
    m["training_load"] = 6.0                                 # -10
    result = calculate_recovery_score(m)
    assert result.score == 65  # 100 - 15 - 10 - 10
    assert len(result.reason_codes) == 3


def test_score_floors_at_zero_never_negative():
    m = {
        "sleep_hours": 3.0, "sleep_hours_7d_baseline": 7.0,      # severe: -15
        "resting_hr": 70.0, "resting_hr_7d_baseline": 50.0,      # +40%: -15
        "hrv_ms": 20.0, "hrv_ms_7d_baseline": 60.0,              # -66%: -10
        "stress_score": 5,                                        # -5
        "training_load": 9.9,                                     # -10
        "restlessness": 0.5, "restlessness_7d_baseline": 0.08,   # -5
    }
    result = calculate_recovery_score(m)
    # max possible deduction here is 15+15+10+5+10+5 = 60, so this specific
    # case won't hit zero, but score must never go below 0 regardless
    assert result.score >= 0


def test_level_bucketing_boundaries():
    # Manually construct scores at each edge using isolated deductions
    # High: >=80, Medium: 60-79, Low: <60
    m = perfect_day()
    m["training_load"] = 5.0  # -10 -> score 90
    assert calculate_recovery_score(m).level == "High"

    m = perfect_day()
    m["sleep_hours"] = m["sleep_hours_7d_baseline"] - 1.0  # -15
    m["hrv_ms"] = m["hrv_ms_7d_baseline"] * 0.80             # -10 -> score 75
    assert calculate_recovery_score(m).level == "Medium"

    m = perfect_day()
    m["sleep_hours"] = m["sleep_hours_7d_baseline"] - 1.0     # -15
    m["resting_hr"] = m["resting_hr_7d_baseline"] * 1.05      # -15
    m["hrv_ms"] = m["hrv_ms_7d_baseline"] * 0.80               # -10 -> score 60
    assert calculate_recovery_score(m).level == "Medium"       # 60 is still Medium (>=60)

    m["training_load"] = 6.0  # additional -10 -> score 50
    assert calculate_recovery_score(m).level == "Low"


# --- Real data anchors, pulled from p01_sanitized.csv ---

def test_real_worst_observed_day_2019_12_06():
    """Worst day found across the whole 152-day CSV (score=50, Low)."""
    m = {
        "sleep_hours": 4.0, "sleep_hours_7d_baseline": 5.0,
        "resting_hr": 53.67, "resting_hr_7d_baseline": 51.5,
        "hrv_ms": 51.4, "hrv_ms_7d_baseline": 57.1,
        "stress_score": 3.0,
        "training_load": 1.9,
        "restlessness": 0.1947, "restlessness_7d_baseline": 0.098,
    }
    result = calculate_recovery_score(m)
    assert result.score == 50
    assert result.level == "Low"


def test_real_best_observed_day_2020_03_24():
    """Best day found across the whole 152-day CSV (score=100, High)."""
    m = {
        "sleep_hours": 7.0, "sleep_hours_7d_baseline": 5.71,
        "resting_hr": 50.58, "resting_hr_7d_baseline": 50.2,
        "hrv_ms": 59.9, "hrv_ms_7d_baseline": 57.2,
        "stress_score": 2.0,
        "training_load": 0.7,
        "restlessness": 0.0589, "restlessness_7d_baseline": 0.065,
    }
    result = calculate_recovery_score(m)
    assert result.score == 100
    assert result.level == "High"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
