"""
grammar.py
----------
Priority 1 feature — three-stage text correction pipeline:

  Stage 1 → Spell correction     (pyspellchecker)
  Stage 2 → Grammar correction   (TextBlob)
  Stage 3 → Capitalisation fix   (first letter of each sentence)

Exposed function:
    correct(text: str) -> dict
"""

from textblob    import TextBlob
from spellchecker import SpellChecker

_spell = SpellChecker()


# ════════════════════════════════════════════════════════════════════════
# Stage 1 — Spell fix
# ════════════════════════════════════════════════════════════════════════

def _fix_spelling(text: str) -> str:
    """
    Replace misspelled words with their most likely correction.
    Preserves original capitalisation and punctuation.
    """
    words     = text.split()
    corrected = []

    for word in words:
        # separate trailing punctuation
        core  = word.rstrip(".,!?;:\"'()")
        trail = word[len(core):]

        if not core:
            corrected.append(word)
            continue

        fix = _spell.correction(core.lower())

        if fix is None:
            corrected.append(word)
            continue

        # restore capitalisation
        if core.isupper():
            fix = fix.upper()
        elif core[0].isupper():
            fix = fix.capitalize()

        corrected.append(fix + trail)

    return " ".join(corrected)


# ════════════════════════════════════════════════════════════════════════
# Stage 2 — Grammar fix
# ════════════════════════════════════════════════════════════════════════

def _fix_grammar(text: str) -> str:
    """Use TextBlob to correct grammar."""
    return str(TextBlob(text).correct())


# ════════════════════════════════════════════════════════════════════════
# Stage 3 — Capitalisation
# ════════════════════════════════════════════════════════════════════════

def _fix_capitalisation(text: str) -> str:
    """Ensure every sentence starts with a capital letter."""
    sentences = text.split(". ")
    return ". ".join(s.strip().capitalize() for s in sentences if s.strip())


# ════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════

def correct(text: str) -> dict:
    """
    Run all three correction stages.

    Returns
    -------
    {
        "original":  str,
        "corrected": str,
        "changed":   bool
    }
    """
    if not text or not text.strip():
        return {"original": text, "corrected": text, "changed": False}

    result = _fix_capitalisation(_fix_grammar(_fix_spelling(text)))

    return {
        "original":  text,
        "corrected": result,
        "changed":   text.strip() != result.strip(),
    }