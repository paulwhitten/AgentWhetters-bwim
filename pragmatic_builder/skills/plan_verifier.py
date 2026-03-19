"""Plan Verifier: post-plan checks that catch common LLM planning errors.

After the LLM planner generates build steps, this module validates them
against the original instruction text to catch systematic errors:

1. Direction consistency — does the plan direction match the instruction?
2. "Each/every" expansion — did the plan expand iterative phrases?
3. No-stacking guard — did horizontal moves become accidental stacks?
4. Block count sanity — does the total block count look reasonable?

Generalises to any instruction — no hard-coded answers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .build_planner import BuildStep

logger = logging.getLogger(__name__)

# Direction words → expected plan direction
_LEFT_WORDS = re.compile(
    r'\b(?:going\s+(?:to\s+)?(?:the\s+)?left|towards?\s+(?:the\s+)?left|to\s+the\s+left'
    r'|going\s+left|leftward)\b',
    re.IGNORECASE,
)
_RIGHT_WORDS = re.compile(
    r'\b(?:going\s+(?:to\s+)?(?:the\s+)?right|towards?\s+(?:the\s+)?right|to\s+the\s+right'
    r'|going\s+right|rightward)\b',
    re.IGNORECASE,
)
_FRONT_WORDS = re.compile(
    r'\b(?:going\s+(?:to\s+)?(?:the\s+)?front|towards?\s+(?:the\s+)?(?:front|bottom)'
    r'|going\s+forward|going\s+front)\b',
    re.IGNORECASE,
)
_BACK_WORDS = re.compile(
    r'\b(?:going\s+(?:to\s+)?(?:the\s+)?back|towards?\s+(?:the\s+)?(?:back|top)'
    r'|going\s+back(?:ward)?)\b',
    re.IGNORECASE,
)

# Phrases that signal "do for each"
_EACH_PATTERN = re.compile(
    r'\b(?:each|every|both|all\s+(?:of\s+)?(?:the|them))\b',
    re.IGNORECASE,
)

# "In front of" / "to the left of" etc. should be HORIZONTAL, not stacking
_HORIZONTAL_REL = re.compile(
    r'\b(?:in\s+front\s+of|to\s+the\s+(?:left|right)\s+of|behind|'
    r'directly\s+(?:in\s+front|to\s+the\s+(?:left|right)|behind))\b',
    re.IGNORECASE,
)

# "On top of" — explicit stacking
_STACK_REL = re.compile(
    r'\b(?:on\s+top\s+of|stack(?:ed)?\s+on)\b',
    re.IGNORECASE,
)

# Number words for count extraction
_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_COUNT_PATTERN = re.compile(
    r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+'
    r'(?:\w+\s+)?(?:blocks?|stack|tower|row)',
    re.IGNORECASE,
)


@dataclass
class VerificationIssue:
    """A single issue found during verification."""
    category: str        # "direction", "each_expansion", "stacking", "count"
    severity: str        # "critical", "warning"
    step_index: int      # which step has the issue (-1 for global)
    description: str
    suggested_fix: str   # instruction for the LLM re-prompt


@dataclass
class VerificationResult:
    """Result of plan verification."""
    issues: list[VerificationIssue] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    def correction_prompt(self) -> str:
        """Build a correction prompt for re-planning."""
        if not self.issues:
            return ""
        lines = ["CORRECTIONS NEEDED (fix these errors in your plan):"]
        for issue in self.issues:
            lines.append(f"- [{issue.category}] {issue.description}. FIX: {issue.suggested_fix}")
        return "\n".join(lines)


def verify_plan(
    instruction: str,
    steps: list[BuildStep],
    existing_block_count: int = 0,
) -> VerificationResult:
    """Verify a build plan against the instruction text.

    Returns a VerificationResult with any issues found.
    """
    result = VerificationResult()

    _check_direction_consistency(instruction, steps, result)
    _check_each_expansion(instruction, steps, result)
    _check_stacking_vs_horizontal(instruction, steps, result)
    _check_block_counts(instruction, steps, existing_block_count, result)

    if result.has_issues:
        logger.info(
            "Plan verification found %d issue(s) (%d critical)",
            len(result.issues),
            sum(1 for i in result.issues if i.severity == "critical"),
        )
        for issue in result.issues:
            logger.info("  [%s] %s: %s", issue.severity, issue.category, issue.description)

    return result


def _check_direction_consistency(
    instruction: str,
    steps: list[BuildStep],
    result: VerificationResult,
) -> None:
    """Verify that extend_row directions match the instruction text."""
    # Find direction words in instruction
    has_left = bool(_LEFT_WORDS.search(instruction))
    has_right = bool(_RIGHT_WORDS.search(instruction))
    has_front = bool(_FRONT_WORDS.search(instruction))
    has_back = bool(_BACK_WORDS.search(instruction))

    for i, step in enumerate(steps):
        if step.action != "extend_row":
            continue
        plan_dir = step.position.get("direction", "").lower()
        if not plan_dir:
            continue

        # Check for contradictions
        if has_left and not has_right and plan_dir == "right":
            result.issues.append(VerificationIssue(
                category="direction",
                severity="critical",
                step_index=i,
                description=(
                    f"Step {i+1} has direction='right' but instruction says "
                    f"'going to the left'. Direction is REVERSED."
                ),
                suggested_fix=(
                    f"Change step {i+1} direction from 'right' to 'left'. "
                    f"'Going to the left' means decreasing x."
                ),
            ))

        if has_right and not has_left and plan_dir == "left":
            result.issues.append(VerificationIssue(
                category="direction",
                severity="critical",
                step_index=i,
                description=(
                    f"Step {i+1} has direction='left' but instruction says "
                    f"'going to the right'. Direction is REVERSED."
                ),
                suggested_fix=(
                    f"Change step {i+1} direction from 'left' to 'right'. "
                    f"'Going to the right' means increasing x."
                ),
            ))

        if has_front and not has_back and plan_dir in ("back", "behind", "backward"):
            result.issues.append(VerificationIssue(
                category="direction",
                severity="critical",
                step_index=i,
                description=(
                    f"Step {i+1} has direction='{plan_dir}' but instruction says "
                    f"'going to the front'. Direction is REVERSED."
                ),
                suggested_fix=(
                    f"Change step {i+1} direction to 'front'. "
                    f"'Going to the front' means increasing z."
                ),
            ))

        if has_back and not has_front and plan_dir in ("front", "forward"):
            result.issues.append(VerificationIssue(
                category="direction",
                severity="critical",
                step_index=i,
                description=(
                    f"Step {i+1} has direction='{plan_dir}' but instruction says "
                    f"'going to the back'. Direction is REVERSED."
                ),
                suggested_fix=(
                    f"Change step {i+1} direction to 'behind'. "
                    f"'Going to the back' means decreasing z."
                ),
            ))


def _check_each_expansion(
    instruction: str,
    steps: list[BuildStep],
    result: VerificationResult,
) -> None:
    """Check that 'each/every/both' phrases are expanded into multiple steps."""
    each_matches = list(_EACH_PATTERN.finditer(instruction))
    if not each_matches:
        return

    # Look for "in front of each X" or "on top of each X" patterns
    # These require one action per referenced block
    for m in each_matches:
        context_start = max(0, m.start() - 50)
        context_end = min(len(instruction), m.end() + 50)
        context = instruction[context_start:context_end]

        # Check if "each" is part of a positioning phrase
        each_pos = re.search(
            r'(?:in\s+front\s+of|on\s+top\s+of|to\s+the\s+(?:left|right)\s+of|behind)'
            r'\s+each\b',
            context, re.IGNORECASE,
        )
        if not each_pos:
            # Also match "one X to each arm/end/corner"
            each_pos = re.search(
                r'\bone\b.*?\bto\s+each\b|\bone\b.*?\bon\s+each\b'
                r'|\beach\s+(?:end|arm|corner|side)\b'
                r'|\beach\s+of\s+(?:the|them)\b'
                r'|\btop\s+of\s+each\b',
                context, re.IGNORECASE,
            )

        if each_pos:
            # Count how many "place" or "stack" steps exist for this part
            # The instruction with "each" should produce multiple steps
            # We can't determine the exact expected count, but check for
            # suspiciously few steps compared to the pattern
            place_steps = [
                s for s in steps
                if s.action in ("place", "place_relative", "stack")
                and s.count in (1, "1")
            ]
            # If "each" appears but only 1 place step exists, that's suspicious
            # (but not always wrong — need context)
            if len(place_steps) < 2 and len(steps) < 4:
                result.issues.append(VerificationIssue(
                    category="each_expansion",
                    severity="warning",
                    step_index=-1,
                    description=(
                        f"Instruction says '{m.group()}' which suggests "
                        f"multiple placements, but plan has fewer steps than expected."
                    ),
                    suggested_fix=(
                        f"Expand the '{m.group()}' phrase into separate steps — "
                        f"one step for each referenced block/position."
                    ),
                ))


def _check_stacking_vs_horizontal(
    instruction: str,
    steps: list[BuildStep],
    result: VerificationResult,
) -> None:
    """Check that horizontal positioning phrases don't become stacks.

    If the instruction says "in front of X" the step should NOT place at
    the same (x,z) as X (which would stack on top).
    """
    # Find all horizontal relationship phrases
    horiz_matches = list(_HORIZONTAL_REL.finditer(instruction))
    if not horiz_matches:
        return

    # Check if any steps share the same (x,z) with a previous step
    # when the instruction has horizontal relationship words
    positions_so_far: dict[tuple[int, int], int] = {}  # (x,z) → step index
    for i, step in enumerate(steps):
        pos = step.position
        if "x" in pos and "z" in pos:
            xz = (int(pos["x"]), int(pos["z"]))
            if xz in positions_so_far and step.action in ("place", "stack", "place_relative"):
                # Same position as a previous step — could be accidental stacking
                prev_i = positions_so_far[xz]
                prev_step = steps[prev_i]
                # Only flag if the instruction has horizontal words near this step's context
                if prev_step.action in ("place", "stack"):
                    result.issues.append(VerificationIssue(
                        category="stacking",
                        severity="warning",
                        step_index=i,
                        description=(
                            f"Step {i+1} ({step.action} {step.color}) places at "
                            f"({xz[0]},{xz[1]}), same as step {prev_i+1} ({prev_step.action} "
                            f"{prev_step.color}). The instruction uses horizontal "
                            f"positioning ('{horiz_matches[0].group()}') — this might be "
                            f"an accidental stack instead of a horizontal placement."
                        ),
                        suggested_fix=(
                            f"The block in step {i+1} should likely be placed at a "
                            f"different (x,z) from step {prev_i+1}. 'In front of' means "
                            f"+z, 'behind' means -z, 'to the right' means +x, "
                            f"'to the left' means -x."
                        ),
                    ))
            positions_so_far[xz] = i


def _check_block_counts(
    instruction: str,
    steps: list[BuildStep],
    existing_block_count: int,
    result: VerificationResult,
) -> None:
    """Check that the total new block count is reasonable."""
    # Extract counts from instruction
    expected_counts = []
    for m in _COUNT_PATTERN.finditer(instruction):
        val_str = m.group(1).lower()
        if val_str.isdigit():
            expected_counts.append(int(val_str))
        elif val_str in _COUNT_WORDS:
            expected_counts.append(_COUNT_WORDS[val_str])

    if not expected_counts:
        return

    # Sum plan blocks
    plan_total = 0
    for step in steps:
        count = step.count
        if isinstance(count, str):
            if count.lower() in ("uncounted", "unknown", "unspecified", "?"):
                continue
            try:
                count = int(count)
            except ValueError:
                continue
        plan_total += count

    expected_total = sum(expected_counts)

    # Allow some tolerance (the instruction may have implicit counts)
    if plan_total < expected_total - 1:
        result.issues.append(VerificationIssue(
            category="count",
            severity="warning",
            step_index=-1,
            description=(
                f"Plan produces {plan_total} new blocks, but instruction "
                f"mentions counts totalling {expected_total}."
            ),
            suggested_fix=(
                f"Review block counts. The instruction specifies: "
                f"{', '.join(str(c) for c in expected_counts)}. "
                f"Ensure the plan accounts for all of them."
            ),
        ))


def auto_fix_direction(
    instruction: str,
    steps: list[BuildStep],
) -> list[BuildStep]:
    """Automatically fix direction errors in extend_row steps.

    This is a deterministic correction — if the instruction says "going
    left" and the plan says "right", just flip it. No LLM needed.
    """
    has_left = bool(_LEFT_WORDS.search(instruction))
    has_right = bool(_RIGHT_WORDS.search(instruction))

    for step in steps:
        if step.action != "extend_row":
            continue
        plan_dir = step.position.get("direction", "").lower()

        if has_left and not has_right and plan_dir == "right":
            logger.info("Auto-fixing direction: right → left for extend_row step")
            step.position["direction"] = "left"

        elif has_right and not has_left and plan_dir == "left":
            logger.info("Auto-fixing direction: left → right for extend_row step")
            step.position["direction"] = "right"

    return steps
