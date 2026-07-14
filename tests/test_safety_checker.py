"""
test_safety_checker.py
-----------------------
Run with:
    pytest tests/test_safety_checker.py -v
"""

import pytest
from safety_checker import check_safety


# --- Medical emergency: should flag and block ---

@pytest.mark.parametrize("text", [
    "I have chest tightness and dizziness. Should I still work out?",
    "I feel dizzy and have chest pain. Should I run?",
    "I think I'm about to faint, is it ok to train?",
    "I can't breathe properly, should I still do cardio?",
    "I have severe breathlessness right now",
    "My heart is racing and I feel light headed",
])
def test_medical_emergency_flags_and_blocks(text):
    result = check_safety(text)
    assert result.flagged is True
    assert result.category == "medical_emergency"
    assert result.blocks_workout_advice is True
    assert result.response_message is not None


# --- Disordered eating / extreme weight loss: should flag and block ---

@pytest.mark.parametrize("text", [
    "I want to lose 5 kg in one week. Make a strict calorie plan.",
    "Help me starve myself to lose weight fast",
    "I want an extreme calorie restriction plan",
    "I've been trying to eat nothing for the past few days",
])
def test_disordered_eating_flags_and_blocks(text):
    result = check_safety(text)
    assert result.flagged is True
    assert result.category == "disordered_eating"
    assert result.blocks_workout_advice is True


# --- Pregnancy: should flag but NOT hard-block ---

def test_pregnancy_flags_but_does_not_block():
    result = check_safety("I'm pregnant, should I train legs today?")
    assert result.flagged is True
    assert result.category == "pregnancy_caution"
    assert result.blocks_workout_advice is False


# --- Priority ordering: emergency should win if multiple categories present ---

def test_medical_emergency_takes_priority_over_pregnancy():
    result = check_safety("I'm pregnant and having chest pain, should I work out?")
    assert result.category == "medical_emergency"
    assert result.blocks_workout_advice is True


# --- Normal queries: should NOT flag ---

@pytest.mark.parametrize("text", [
    "My sleep duration was 7 hours, but I feel exhausted today.",
    "Should I do leg day today?",
    "My resting heart rate is higher than usual and my sleep was poor.",
    "My stress score is high today. What workout should I do?",
    "Can you update my workout plan for this week?",
])
def test_normal_queries_not_flagged(text):
    result = check_safety(text)
    assert result.flagged is False
    assert result.category is None
    assert result.blocks_workout_advice is False


# --- False-positive guard: everyday fitness phrasing shouldn't trip anything ---

@pytest.mark.parametrize("text", [
    "I did leg day earlier and now I feel stronger",
    "I ran faster than my usual pace today",
    "I feel a bit tired but nothing serious, just want a lighter workout",
])
def test_common_fitness_phrasing_no_false_positive(text):
    result = check_safety(text)
    assert result.flagged is False


# --- Edge cases ---

def test_empty_string_not_flagged():
    result = check_safety("")
    assert result.flagged is False
    assert result.category is None


def test_whitespace_only_not_flagged():
    result = check_safety("   ")
    assert result.flagged is False


def test_case_insensitivity():
    result = check_safety("I HAVE CHEST PAIN AND DIZZINESS")
    assert result.flagged is True
    assert result.category == "medical_emergency"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
