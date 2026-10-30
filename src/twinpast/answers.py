"""Deterministic, gold-free extraction of explicit multiple-choice answers."""
import re
import unicodedata

from .data import require

PARSER_VERSION = "explicit_option_v1"
READOUTS = ("explicit_option", "exact_letter", "first_standalone_letter")

# A marker must start a line or sentence, optionally after a conclusion cue.
# This excludes mentions such as "I considered the answer A" and negated choices.
_MARKER = re.compile(
    r"(?:^|\n|(?<=[.!?])\s+)[ \t]*(?:(?:therefore|thus|so)[, ]+)?"
    r"(?:the\s+)?(?P<final>final\s+)?(?:(?:correct|best)\s+)?"
    r"(?:answer|option|choice)(?:\s+is)?\s*[:=]?\s*", re.I)
_UNCERTAIN = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|either|unsure|uncertain)\b|"
    r"\b(?:not sure|cannot decide|can't decide|might be|could be)\b", re.I)


def _clean(text):
    text = unicodedata.normalize("NFKC", text).strip()
    # Only known end-of-turn tokens; silence and other special tokens stay invalid.
    text = re.sub(r"(?:\s*<\|(?:im_end|endoftext|eot_id)\|>)+\s*$", "", text)
    text = re.sub(r"^\(Video Time\s*=\s*\d+(?:\.\d+)?s\)\s*Assistant\s*:\s*", "", text)
    text = re.sub(r"^(?:(?:<\|(?:response|round_start)\|>|</?response>)\s*)+", "", text)
    text = re.sub(r"\s*</response>\s*$", "", text)
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            return ""
        text = text[end + len("</think>"):].strip()
    fence = re.fullmatch(r"```(?:text|plaintext)?\s*\n(.*?)\n?```", text, re.S | re.I)
    if fence:
        text = fence[1].strip()
    # Remove paired Markdown emphasis, preserving literal underscores in content.
    for pattern in (r"\*\*(.+?)\*\*", r"__(.+?)__", r"`([^`]+)`", r"\*([^*\n]+)\*"):
        text = re.sub(pattern, r"\1", text, flags=re.S)
    return re.sub(r"(?m)^[ \t]*#{1,6}\s+", "", text).strip()


def _normalized(text):
    # Preserve internal punctuation, negation, numbers and event order. No fuzzy
    # substring matching, stemming, counting or semantic inference is performed.
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\s*([,;:])\s*", r"\1", text)
    return " ".join(text.split()).strip(" \"'“”‘’").rstrip(".,!?;:。)")


def _choice(text, options):
    """Read a complete answer, optionally followed by a marked explanation."""
    text = text.strip()
    if text.endswith("?"):
        return None
    value = _normalized(text)
    matches = [i for i, option in enumerate(options) if value and value == _normalized(option)]
    if matches:
        return chr(65 + matches[0]) if len(matches) == 1 else None

    # An explicit rationale does not become a source of candidate letters.
    parts = re.split(r"\s+because\b|\n\s*(?:explanation|reason)\s*:", text, maxsplit=1, flags=re.I)
    if len(parts) == 2:
        return _choice(parts[0].rstrip(" ,;:"), options)
    if _UNCERTAIN.search(text):
        return None

    boxed = re.fullmatch(r"\\boxed\{([^{}]+)\}[.!]?", text)
    if boxed:
        return _choice(boxed[1], options)
    bare = re.fullmatch(r"[\s\"'“”‘’(\[]*([A-Za-z])[\s\"'“”‘’)\].!,:;。]*", text)
    if bare:
        letter = bare[1].upper()
        return letter if ord(letter) < 65 + len(options) else None

    # A letter accompanied by content is accepted only when that complete
    # content uniquely matches the SAME option. Contradictions stay unresolved.
    labeled = re.fullmatch(r"\(?([A-Za-z])\)?\s*[.):,\-–—]\s*(\S.*)", text, re.S)
    if labeled:
        letter = labeled[1].upper()
        content = _normalized(labeled[2])
        matches = [i for i, option in enumerate(options) if content and content == _normalized(option)]
        if len(matches) == 1 and chr(65 + matches[0]) == letter:
            return letter
    return None


def parse_answer(text, options, readout="explicit_option"):
    """Return one option letter or None, using only the response and options.

    A distinct final-answer marker overrides earlier discussion/option lists.
    Multiple selected answer statements must agree and must all be resolvable.
    The two legacy modes retain their original behavior for explicit replay.
    """
    require(readout in READOUTS, "Unknown readout")
    if not isinstance(text, str):
        return None
    labels = "".join(chr(65 + i) for i in range(len(options)))
    if readout == "exact_letter":
        value = text.strip()
        return value if len(value) == 1 and value in labels else None
    if readout == "first_standalone_letter":
        match = re.search(r"(?<![A-Z0-9_])([" + labels + r"])(?![A-Z0-9_])", text.strip().upper())
        return match[1] if match else None

    text = _clean(text)
    if not text:
        return None
    markers = list(_MARKER.finditer(text))
    if markers:
        first_final = next((i for i, m in enumerate(markers) if m["final"]), 0)
        answers = []
        for index, marker in enumerate(markers):
            if index < first_final:
                continue
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            answers.append(_choice(text[marker.end():end], options))
        return answers[0] if answers[0] is not None and len(set(answers)) == 1 else None
    answer = _choice(text, options)
    if answer is not None:
        return answer
    # A complete answer in its own final paragraph is a commitment, unlike an
    # arbitrary letter found inside a sentence. Reject competing answer blocks.
    paragraphs = re.split(r"\n[ \t]*\n", text)
    if len(paragraphs) > 1:
        answer = _choice(paragraphs[-1], options)
        earlier = [_choice(line, options) for p in paragraphs[:-1] for line in p.splitlines()]
        if answer is not None and all(p is None or p == answer for p in earlier):
            return answer
    return None
