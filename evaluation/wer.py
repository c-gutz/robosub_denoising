"""Word error rate between reference transcripts and ASR output.

WER = (substitutions + deletions + insertions) / reference words. Lower is
better; it can exceed 1.0 when the ASR inserts many extra words.
"""

import re
from collections.abc import Callable


def basic_normalize(text: str) -> str:
    """Lowercase and strip punctuation so formatting isn't counted as errors."""
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return " ".join(text.split())


def wer(
    references: str | list[str],
    hypotheses: str | list[str],
    normalize: Callable[[str], str] = basic_normalize,
) -> dict[str, float]:
    """Score one transcript, or a list of them pooled into one corpus-level WER."""
    import jiwer

    if isinstance(references, str):
        references, hypotheses = [references], [hypotheses]
    out = jiwer.process_words(
        [normalize(r) for r in references],
        [normalize(h) for h in hypotheses],
    )
    return {
        "wer": out.wer,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
    }
