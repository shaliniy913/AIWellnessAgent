"""
safety_checker.py
------------------
Deterministic keyword/pattern-based safety guardrail. No LLM involved on
purpose — safety gating should be predictable and auditable, not left to a
model's judgment call. This is intentionally simple per the project brief's
"manual safety rules file" suggestion (section 19).

Usage:
    from safety_checker import check_safety

    result = check_safety(user_text)
    if result.flagged:
        # do not proceed to wearable analysis / workout recommendation
        return result.response_message

Limitations (documented, not hidden):
    - Plain substring/regex matching. No negation handling — "I don't have
      chest pain" will still flag, because reliably parsing negation is a
      much harder NLP problem than this capstone needs, and erring toward
      over-flagging is the safer failure mode for a wellness app.
    - Not a medical triage tool. It exists only to stop the app from giving
      workout advice when it shouldn't, and to hand off to a human/
      professional resource instead.
"""

import re
from dataclasses import dataclass, field


@dataclass
class SafetyCheckResult:
    flagged: bool
    category: str | None          # "medical_emergency" | "disordered_eating" | "pregnancy_caution" | None
    matched_terms: list = field(default_factory=list)
    response_message: str | None = None
    blocks_workout_advice: bool = False  # True = stop pipeline here entirely


# ---------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------
# Each pattern is matched case-insensitively with word boundaries where it
# matters. Multi-word phrases don't need \b on every word — the phrase
# itself is specific enough.

MEDICAL_EMERGENCY_PATTERNS = [
    r"chest pain", r"chest tightness", r"chest pressure",
    r"\bdizz(y|iness)\b", r"light\s?headed",
    r"\bfaint(ing|ed)?\b", r"blacking out", r"passed out",
    r"can'?t breathe", r"shortness of breath", r"severe breathlessness",
    r"irregular heartbeat", r"heart palpitations", r"racing heart",
    r"numbness in (my |the )?(arm|face|leg)",
    r"severe injury", r"can'?t move (my |the )?(arm|leg|shoulder)",
    r"severe dehydration",
    r"sudden severe fatigue",
    r"medical emergency", r"call(ing)? (an )?ambulance",
    r"emergency room", r"\ber\b\s+(now|visit|immediately)", r"call 911", r"call (the )?paramedics",
]

DISORDERED_EATING_PATTERNS = [
    r"lose\s+\d+\s*(kg|kilo|kilograms|lbs?|pounds?)\s+in\s+\d+\s*(day|days|week|weeks)",
    r"extreme(ly)? (calorie|weight)",
    r"starve (myself|for)", r"\bstarving myself\b",
    r"eat (nothing|almost nothing|zero calories)",
    r"skip all meals", r"stop eating",
    r"crash diet",
    r"purge(ing)? after (eating|meals)",
    r"binge and purge",
    r"strict calorie plan",
]

PREGNANCY_PATTERNS = [
    r"\bpregnan(t|cy)\b",
]


def _find_matches(text: str, patterns: list) -> list:
    matches = []
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            matches.append(p)
    return matches


def check_safety(text: str) -> SafetyCheckResult:
    """
    Check user input for safety-relevant language. Checked in priority
    order: medical emergency > disordered eating > pregnancy caution.
    Only the highest-priority match is returned, since a medical emergency
    flag should always take precedence and short-circuit the response.
    """
    if not text or not text.strip():
        return SafetyCheckResult(flagged=False, category=None)

    emergency_matches = _find_matches(text, MEDICAL_EMERGENCY_PATTERNS)
    if emergency_matches:
        return SafetyCheckResult(
            flagged=True,
            category="medical_emergency",
            matched_terms=emergency_matches,
            blocks_workout_advice=True,
            response_message=(
                "I can't recommend a workout based on what you've described. "
                "Please avoid exercising right now and speak to a qualified "
                "medical professional — if symptoms are severe, seek urgent "
                "care. I'm a wellness assistant, not a medical service, and "
                "this isn't something I can advise on further."
            ),
        )

    eating_matches = _find_matches(text, DISORDERED_EATING_PATTERNS)
    if eating_matches:
        return SafetyCheckResult(
            flagged=True,
            category="disordered_eating",
            matched_terms=eating_matches,
            blocks_workout_advice=True,
            response_message=(
                "I can't create an extreme or rapid weight-loss / restrictive "
                "eating plan — that kind of approach can be harmful. I can "
                "offer general wellness guidance on sustainable habits "
                "instead, and if this is something you're struggling with, "
                "please consider speaking with a doctor or registered "
                "dietitian."
            ),
        )

    pregnancy_matches = _find_matches(text, PREGNANCY_PATTERNS)
    if pregnancy_matches:
        return SafetyCheckResult(
            flagged=True,
            category="pregnancy_caution",
            matched_terms=pregnancy_matches,
            blocks_workout_advice=False,  # caution, not a hard stop
            response_message=(
                "Since this involves pregnancy, I'll keep this to general "
                "wellness information only — pregnancy-specific exercise "
                "guidance should come from your doctor or midwife, since "
                "safe activity varies a lot by trimester and individual "
                "health. I can still share general recovery signals from "
                "your data if that's helpful."
            ),
        )

    return SafetyCheckResult(flagged=False, category=None)


if __name__ == "__main__":
    tests = [
        "My sleep duration was 7 hours, but I feel exhausted today.",
        "I have chest tightness and dizziness. Should I still work out?",
        "I want to lose 5 kg in one week. Make a strict calorie plan.",
        "Should I do leg day today?",
        "I'm pregnant and want to know if I should train legs today.",
    ]
    for t in tests:
        r = check_safety(t)
        print(f"\nInput: {t}")
        print(f"  flagged={r.flagged} category={r.category} blocks={r.blocks_workout_advice}")
        if r.response_message:
            print(f"  -> {r.response_message}")
