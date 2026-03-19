"""Detect underspecified instructions (missing color or number).

Strategy:
- Missing NUMBER is never worth asking about — use a heuristic (match
  the height of a referenced stack, or default to 3).
- Missing COLOR is only worth asking if the color is genuinely
  unpredictable — i.e., a block-placing phrase has no color and we
  cannot infer it from any color mentioned elsewhere in the instruction.
  If the instruction mentions exactly one color, we can reuse it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .grid import VALID_COLORS

logger = logging.getLogger(__name__)

# Color words that may appear in instructions (lowercase for matching)
_COLOR_WORDS = {c.lower() for c in VALID_COLORS}

# Map of word numbers to ints
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


@dataclass
class UnderspecResult:
    """Result of underspecification detection."""
    has_missing_color: bool = False
    has_missing_number: bool = False
    suggested_question: str = ""
    details: str = ""
    # Colors found in the instruction that can be used for inference
    colors_in_instruction: list[str] = field(default_factory=list)
    # Suggested default color if we decide not to ask
    inferred_color: str = ""
    # Suggested default count if we decide not to ask
    inferred_count: int = 0


def _extract_colors_from_text(text: str) -> list[str]:
    """Extract all color words from text in order of appearance."""
    colors = []
    for m in re.finditer(r'\b(' + '|'.join(_COLOR_WORDS) + r')\b', text.lower()):
        colors.append(m.group(1).capitalize())
    return colors


def _count_block_placing_phrases(text: str) -> list[dict]:
    """Parse block-placing phrases and determine which have/lack color and count."""
    text_lower = text.lower()
    phrases = []

    # Match phrases like "stack N color blocks", "build a color stack",
    # "place two blocks", "stack blocks", etc.
    patterns = [
        # "stack/place/build/add/put [N] [color] blocks/stack/tower"
        r'(?:stack|place|build|add|put|extend)\s+'
        r'(?:a\s+)?'
        r'(?:(?:tower|row|stack|line|horizontal\s+row)\s+of\s+)?'
        r'(?:(?:\d+|one|two|three|four|five|six|seven|eight|nine)\s+)?'
        r'(?:(?:' + '|'.join(_COLOR_WORDS) + r')\s+)?'
        r'(?:blocks?|stack|tower|row)\b',
        # "build a [color] stack/tower"
        r'(?:build|finish\s+with)\s+(?:a\s+)?'
        r'(?:(?:' + '|'.join(_COLOR_WORDS) + r')\s+)?'
        r'(?:stack|tower)',
    ]

    for pattern in patterns:
        for m in re.finditer(pattern, text_lower):
            phrase = m.group()
            has_color = any(c in phrase for c in _COLOR_WORDS)
            has_number = bool(re.search(
                r'\b(?:\d+|one|two|three|four|five|six|seven|eight|nine)\b', phrase
            ))
            phrases.append({
                'phrase': phrase,
                'has_color': has_color,
                'has_number': has_number,
                'start': m.start(),
                'end': m.end(),
            })

    return phrases


def detect_underspec_heuristic(instruction: str) -> UnderspecResult:
    """Detect missing colors or numbers using regex heuristics.

    Only flags genuinely missing colors that cannot be inferred.
    Never flags missing numbers — those are resolved with heuristics.
    """
    result = UnderspecResult()
    text = instruction.lower()

    # Extract all colors mentioned anywhere in the instruction
    all_colors = _extract_colors_from_text(text)
    result.colors_in_instruction = all_colors
    unique_colors = list(dict.fromkeys(all_colors))  # dedupe, preserve order

    # Parse block-placing phrases
    phrases = _count_block_placing_phrases(text)
    colored = [p for p in phrases if p['has_color']]

    # Check for phrases without color — but filter out sub-phrases that
    # are contained within a longer colored phrase (e.g., "build a stack"
    # inside "build a stack of three blue blocks") and phrases whose
    # lookahead context already contains a color word.
    colorless_phrases = [p for p in phrases if not p['has_color']]

    # Filter: remove sub-phrases overlapping with colored phrases
    def _overlaps(cp: dict) -> bool:
        for col in colored:
            if cp['start'] >= col['start'] and cp['end'] <= col['end']:
                return True
            if col['start'] >= cp['start'] and col['end'] <= cp['end']:
                return True
        return False

    colorless_phrases = [p for p in colorless_phrases if not _overlaps(p)]

    # Filter: skip phrases whose 30-char lookahead already has a color
    def _context_has_color(p: dict) -> bool:
        lookahead = text[p['end']:p['end'] + 30]
        for c in _COLOR_WORDS:
            if re.search(r'\b' + c + r'\b', lookahead):
                return True
        return False

    colorless_phrases = [p for p in colorless_phrases if not _context_has_color(p)]

    if colorless_phrases:
        if len(unique_colors) == 0:
            # No colors at all in the instruction — truly unknown
            result.has_missing_color = True
            result.details = "No color words found in instruction at all. "
            result.suggested_question = "What color should the unspecified blocks be?"
        elif len(unique_colors) == 1:
            # Single color mentioned but some phrases lack color.
            # Always ask — at 85% build success, asking (+5 net) beats
            # inference (~54% correct → expected -10×0.46 + 10×0.54 = +0.8).
            # Keep inferred_color as fallback for the answer-path patcher.
            result.has_missing_color = True
            result.inferred_color = unique_colors[0]
            result.details = (
                f"Single color '{unique_colors[0]}' but colorless phrases present. "
                f"Asking is +EV vs inference. "
            )
            result.suggested_question = (
                "What color should the unspecified blocks be?"
            )
        else:
            # Multiple colors mentioned but some phrases lack color.
            # The missing color is genuinely unpredictable — ASK.
            # Data shows color_under targets always use a color NOT in the
            # instruction, so guessing is ~0% accurate.  Asking costs -5
            # but correct answer earns +10 → net +5 beats guessing (-10).
            result.has_missing_color = True
            result.details = (
                f"Multiple colors ({unique_colors}) but some phrases lack color. "
                f"Must ask — cannot reliably infer. "
            )
            result.suggested_question = (
                "What color should the unspecified blocks be?"
            )

    # Check for phrases without number — NEVER ask, just note for heuristic
    numberless_phrases = [p for p in phrases if not p['has_number']]
    if numberless_phrases:
        result.has_missing_number = True
        # Default: match the count of the first specified stack, or use 3
        # Extract the first explicit count from the instruction
        count_match = re.search(
            r'\b(?:(\d+)|(one|two|three|four|five|six|seven|eight|nine))\b', text
        )
        if count_match:
            if count_match.group(1):
                result.inferred_count = int(count_match.group(1))
            else:
                result.inferred_count = _WORD_NUMBERS.get(count_match.group(2), 3)
        else:
            result.inferred_count = 3  # reasonable default
        result.details += (
            f"Missing count detected, inferred as {result.inferred_count} "
            f"(matched from instruction or default). "
        )
        # Note: has_missing_number is set but NO suggested_question —
        # the pipeline should use inferred_count instead of asking.

    return result


def patch_instruction_with_color(instruction: str, color: str) -> str:
    """Insert a color word into colorless block-placing phrases.

    Used to patch the instruction BEFORE sending to the LLM, so the planner
    receives a fully-specified instruction and doesn't have to guess colors.

    Examples:
        >>> patch_instruction_with_color(
        ...     "Build a tower of four blocks in front", "Blue")
        'Build a tower of four blue blocks in front'
        >>> patch_instruction_with_color(
        ...     "Then place a block on top", "Red")
        'Then place a red block on top'
    """
    phrases = _count_block_placing_phrases(instruction)
    colored = [p for p in phrases if p['has_color']]
    colorless = [p for p in phrases if not p['has_color']]

    if not colorless:
        return instruction

    # Remove colorless phrases that overlap with a colored phrase
    def _overlaps_colored(cp: dict) -> bool:
        for col in colored:
            if cp['start'] >= col['start'] and cp['end'] <= col['end']:
                return True
            if col['start'] >= cp['start'] and col['end'] <= cp['end']:
                return True
        return False

    colorless = [p for p in colorless if not _overlaps_colored(p)]

    if not colorless:
        return instruction

    # Skip phrases whose immediate context already contains a color
    # e.g. "build a tower" when the full clause is "build a tower of three blue blocks"
    def _context_has_color(p: dict) -> bool:
        lookahead = instruction[p['end']:p['end'] + 30].lower()
        for c in _COLOR_WORDS:
            if re.search(r'\b' + c + r'\b', lookahead):
                return True
        return False

    colorless = [p for p in colorless if not _context_has_color(p)]

    if not colorless:
        return instruction

    # Deduplicate overlapping — keep the longest match at each start position
    by_start: dict[int, dict] = {}
    for p in colorless:
        s = p['start']
        if s not in by_start or (p['end'] - p['start']) > (by_start[s]['end'] - by_start[s]['start']):
            by_start[s] = p
    unique = sorted(by_start.values(), key=lambda p: p['start'], reverse=True)

    # Insert color word right before the final noun (blocks/block/stack/tower/row)
    # Process right-to-left so earlier positions stay valid
    result = instruction
    color_lower = color.lower()
    for p in unique:
        phrase_text = result[p['start']:p['end']]
        m = re.search(r'\b(blocks?|stack|tower|row)\s*$', phrase_text, re.IGNORECASE)
        if m:
            insert_pos = p['start'] + m.start()
            result = result[:insert_pos] + color_lower + ' ' + result[insert_pos:]

    return result


def detect_underspec_from_plan(plan_steps: list[dict]) -> UnderspecResult:
    """Detect underspecification from LLM plan output.

    Checks if the LLM output contains 'Uncolored' or 'Uncounted' placeholders.
    For 'Uncounted', provides a heuristic count instead of flagging to ask.
    For 'Uncolored', only flags if genuinely unpredictable.
    """
    result = UnderspecResult()

    # Collect all specified colors from the plan for inference
    plan_colors = []
    plan_counts = []
    for step in plan_steps:
        color = str(step.get("color", ""))
        count = step.get("count", "")
        if color.lower() not in ("uncolored", "unknown", "unspecified", "?", ""):
            plan_colors.append(color.capitalize())
        if isinstance(count, int) or (isinstance(count, str) and count.isdigit()):
            plan_counts.append(int(count))

    unique_plan_colors = list(dict.fromkeys(plan_colors))

    has_uncolored = False
    has_uncounted = False

    for step in plan_steps:
        color = str(step.get("color", ""))
        count = step.get("count", "")

        if color.lower() in ("uncolored", "unknown", "unspecified", "?"):
            has_uncolored = True

        if str(count).lower() in ("uncounted", "unknown", "unspecified", "?"):
            has_uncounted = True

    # ── Color resolution ──
    if has_uncolored:
        if len(unique_plan_colors) == 1:
            # Only one color in the plan — safe to infer
            result.inferred_color = unique_plan_colors[0]
            result.details += (
                f"LLM flagged unspecified color, inferring "
                f"'{unique_plan_colors[0]}' (only color in plan). "
            )
        else:
            # Multiple colors OR no colors — cannot reliably guess.
            # Always ASK.  Guessing is ~0% accurate on color_under trials.
            result.has_missing_color = True
            result.details += (
                "LLM flagged unspecified color, must ask "
                f"(plan colors: {unique_plan_colors}). "
            )

    # ── Count resolution ──
    if has_uncounted:
        result.has_missing_number = True
        if plan_counts:
            result.inferred_count = plan_counts[-1]
        else:
            result.inferred_count = 3
        result.details += (
            f"LLM flagged unspecified count, inferred as "
            f"{result.inferred_count}. "
        )

    # ── Build question — only ask about COLOR, never about count ──
    # The green agent responds with a single color word; asking about count
    # would waste the one allowed question.  Counts are always inferred.
    if result.has_missing_color:
        result.suggested_question = (
            "What color should the unspecified blocks be?"
        )
    # Note: count-only underspec does NOT generate a question —
    # we always infer counts via heuristic (never ask).

    return result
