"""
rewriter.py
-----------
Rewrites / improves portfolio text using rule-based NLP techniques:

  - Removes filler and weak phrases
  - Strengthens action verbs
  - Improves sentence structure
  - Makes text more professional and concise

No external AI API required — runs fully locally.

Exposed function:
    rewrite(text: str, field: str) -> dict
"""

import re

# ── Weak phrase replacements ─────────────────────────────────────────────
WEAK_PHRASES = {
    r"\bi am responsible for\b":       "I manage",
    r"\bi was responsible for\b":      "I managed",
    r"\bi helped (to )?":              "I ",
    r"\bi tried to\b":                 "I",
    r"\bi worked on\b":                "I developed",
    r"\bi did\b":                      "I delivered",
    r"\bi made\b":                     "I built",
    r"\bi was involved in\b":          "I contributed to",
    r"\bi have experience in\b":       "I am experienced in",
    r"\bi have knowledge of\b":        "I am proficient in",
    r"\bi know\b":                     "I am skilled in",
    r"\bvery good\b":                  "strong",
    r"\bgood\b":                       "proficient",
    r"\bvery\b":                       "",
    r"\breally\b":                     "",
    r"\bbasically\b":                  "",
    r"\bkind of\b":                    "",
    r"\bsort of\b":                    "",
    r"\bin order to\b":                "to",
    r"\bdue to the fact that\b":       "because",
    r"\bat this point in time\b":      "currently",
    r"\bon a daily basis\b":           "daily",
    r"\bhas the ability to\b":         "can",
    r"\bam able to\b":                 "can",
}

# ── Action verb upgrades ─────────────────────────────────────────────────
ACTION_UPGRADES = {
    r"\bworked with\b":     "collaborated with",
    r"\bused\b":            "leveraged",
    r"\bhelped\b":          "supported",
    r"\bdid\b":             "executed",
    r"\bmade\b":            "engineered",
    r"\bfixed\b":           "resolved",
    r"\bchanged\b":         "optimised",
    r"\bshowed\b":          "demonstrated",
    r"\btalked to\b":       "communicated with",
    r"\bwent through\b":    "reviewed",
    r"\bput together\b":    "assembled",
    r"\bset up\b":          "configured",
    r"\bstarted\b":         "initiated",
    r"\bfinished\b":        "completed",
    r"\bgot\b":             "achieved",
}

# ── Field-specific improvement hints ─────────────────────────────────────
FIELD_HINTS = {
    "bio":        "Start with a strong identity statement. Focus on value you bring.",
    "skills":     "List skills separated by commas. Group related technologies together.",
    "experience": "Use past-tense action verbs. Quantify achievements where possible.",
    "education":  "Include degree, institution, and year. Add certifications if any.",
    "projects":   "Describe what you built, the tech stack used, and the impact or outcome.",
}


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════

def _apply_replacements(text: str, replacements: dict) -> str:
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # clean up double spaces
    text = re.sub(r" {2,}", " ", text).strip()
    return text


def _fix_sentence_endings(text: str) -> str:
    """Ensure each sentence ends with a period."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return ". ".join(sentences) + ("." if sentences else "")


def _capitalize_sentences(text: str) -> str:
    sentences = text.split(". ")
    return ". ".join(s.strip().capitalize() for s in sentences if s.strip())


# ════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════

def rewrite(text: str, field: str = "bio") -> dict:
    """
    Rewrite and improve portfolio text for the given field.

    Parameters
    ----------
    text  : str — original text from the form field
    field : str — one of: bio, skills, experience, education, projects

    Returns
    -------
    {
        "original":  str,
        "rewritten": str,
        "changed":   bool,
        "hint":      str   — field-specific writing tip
    }
    """
    if not text or not text.strip():
        return {
            "original":  text,
            "rewritten": text,
            "changed":   False,
            "hint":      FIELD_HINTS.get(field.lower(), ""),
        }

    improved = text

    # Stage 1 — remove weak phrases
    improved = _apply_replacements(improved, WEAK_PHRASES)

    # Stage 2 — upgrade action verbs
    improved = _apply_replacements(improved, ACTION_UPGRADES)

    # Stage 3 — fix sentence structure
    improved = _fix_sentence_endings(improved)
    improved = _capitalize_sentences(improved)

    return {
        "original":  text,
        "rewritten": improved,
        "changed":   text.strip() != improved.strip(),
        "hint":      FIELD_HINTS.get(field.lower(), ""),
    }


# ── Add new field hints for the 4 new fields ─────────────────────────────
FIELD_HINTS.update({
    "introduction":          "Start with a confident opener. State your name, field, and what makes you unique.",
    "additional_experience": "Include volunteering, hackathons, clubs, research, and open source work.",
    "qualifications":        "List certifications with the issuing body and year. One per line for clarity.",
})