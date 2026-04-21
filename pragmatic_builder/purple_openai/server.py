import argparse
import asyncio
import logging
import os
import re
import sys
import time
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
import httpx
from openai import AsyncOpenAI, AsyncAzureOpenAI, APITimeoutError, APIConnectionError
import uvicorn

# Add parent directory to path so skills package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.vllm_compat import is_vllm_mode, strip_think_tags, vllm_extra_body
from skills.instruction_parser import parse_green_message, ParsedInstruction
from skills.build_planner import BuildPlanner
from skills.spatial_executor import SpatialExecutor, ExecutionError
from skills.underspec_detector import detect_underspec_heuristic, patch_instruction_with_color, patch_instruction_with_count
from skills.response_formatter import format_build_response, validate_build_response
from skills.grid import Grid, GridConfig
from skills.structure_analyzer import analyze_structure
from skills.plan_verifier import verify_plan, auto_fix_direction, auto_fix_each_end_caps, auto_fix_t_shape_extend
from skills.plan_patcher import patch_chain_references

logger = logging.getLogger(__name__)

# Direct LLM fallback system prompt (used when skills pipeline fails)
_FALLBACK_SYSTEM_PROMPT = (
    "You are a block-building agent on a 9x9 grid.\n\n"

    "GRID COORDINATES:\n"
    "- The grid is the x-z plane. Origin (0,0) is the center.\n"
    "- Valid x,z coordinates: [-400,-300,-200,-100,0,100,200,300,400]\n"
    "- Y-axis is vertical (height). Ground level y=50. Each block adds +100.\n"
    "- Valid y coordinates: [50,150,250,350,450]\n"
    "- Format: Color,x,y,z (e.g., Red,0,50,0 means a red block at center, ground level)\n\n"

    "DIRECTIONS (CRITICAL):\n"
    "- 'in front of' = +z direction (increasing z)\n"
    "- 'behind' = -z direction (decreasing z)\n"
    "- 'to the right' = +x direction (increasing x)\n"
    "- 'to the left' = -x direction (decreasing x)\n"
    "- 'on top of' = +y direction (increasing y)\n\n"

    "CORNERS:\n"
    "- bottom left = (-400, 50, 400), bottom right = (400, 50, 400)\n"
    "- top left = (-400, 50, -400), top right = (400, 50, -400)\n\n"

    "YOUR RESPONSE FORMAT:\n"
    "You must respond with ONLY this format:\n\n"

    "[BUILD];Color,x,y,z;Color,x,y,z;...\n"
    "- List ALL blocks that should be on the grid (existing + new)\n"
    "- No spaces, semicolons separate blocks\n"
    "- Colors capitalized (Red, Blue, Green, Yellow, Purple, etc.)\n"
    "- NEVER respond with [ASK] — always BUILD your best guess.\n\n"

    "STRATEGY:\n"
    "- If a color is not specified, pick the most likely color from context\n"
    "  (e.g., reuse the only color mentioned, or match adjacent blocks).\n"
    "- If a NUMBER of blocks is not specified, GUESS based on context:\n"
    "  * 'Stack blocks in front' with no count → match the height of the\n"
    "    referenced stack.\n"
    "  * 'Build a stack to the left' → match the adjacent stack height.\n"
    "  * When in doubt, use 3 blocks.\n"
    "- Include START_STRUCTURE blocks plus new blocks in [BUILD].\n\n"

    "SPATIAL RULES (many errors come from these):\n"
    "- 'highlighted square' / 'middle square' = origin (0,0). A row 'starting from\n"
    "  the middle square going right' starts AT x=0, NOT x=100.\n"
    "- 'each end' of a row = leftmost AND rightmost. After extending, recalculate!\n"
    "  E.g., row at [-100,0,100] extended by 2 right → ends are -100 and 300.\n"
    "- 'in front of the leftmost' = find the block with MIN x, then +z.\n"
    "- Double-check direction: 'going left' = decreasing x, 'going right' = +x.\n"
    "- 'Nine blocks along left edge' = 9 blocks at x=-400 varying z from -400..400.\n"
    "- 'to the left/right of X' = HORIZONTAL move (change x), NOT stacking on top.\n"
    "  'Red left of green at (0,0), yellow left of red' → red(-100,0), yellow(-200,0).\n"
    "- Chain references: 'stack green behind existing at (0,0)' → green at (0,-100).\n"
    "  'yellow to right of the green one' → yellow at (100,-100), NOT (100,0).\n"
    "- L/T shapes: 'extend the longer side' = add horizontally in that direction,\n"
    "  NOT stack on top.\n"
)


def prepare_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="block_building",
        name="Block building",
        description="Build block on the grid using spatial reasoning skills",
        tags=["blocks", "building", "spatial"],
        examples=[],
    )
    return AgentCard(
        name="AgentWhetters_BWIM",
        description="Spatial reasoning agent from AgentWhetters, powered by gpt-4o-mini.",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


# Default timeout for LLM requests (seconds).  Thor under load can take
# 60-120s per request; set high to avoid premature timeouts.
_LLM_TIMEOUT = float(
    os.getenv("OPENAI_TIMEOUT_PURPLE", "").strip()
    or os.getenv("OPENAI_TIMEOUT", "600")
)

# How many times to retry a transient LLM error (timeout / connection reset).
_LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# Transient exception types from httpx / openai that are worth retrying.
_TRANSIENT_ERRORS = (
    APITimeoutError,
    APIConnectionError,
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)


def _make_openai_client(api_key: str, base_url: str | None = None) -> AsyncOpenAI:
    """Create an OpenAI client, using Azure or GitHub Models when configured."""
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    if azure_endpoint:
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        return AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=_LLM_TIMEOUT,
        )
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=_LLM_TIMEOUT)


class OpenAIPurpleAgent(AgentExecutor):
    """Purple agent with skills-based build pipeline.

    Pipeline: parse -> plan (LLM) -> detect underspec -> execute (deterministic) -> format
    Falls back to direct LLM call if the skills pipeline fails at any stage.

    When a color is genuinely under-specified (the instruction uses some colors but
    leaves one or more unspecified), the agent ASKs for the missing color.  After
    receiving the answer it re-runs execution with the clarified color and BUILDs.
    Asking costs -5 but getting it right earns +10, so net +5 beats guessing (~0 EV).

    When a count is missing but color is known, the agent ASKs for the missing
    count.  Heuristic count inference is only 64.6% accurate, so asking (EV +5.0)
    dominates guessing (EV +2.9).  Only one question per round; color has priority.
    """

    def __init__(self, debug: bool = False):
        self._debug = debug
        # _PURPLE vars let the purple agent use a different backend than the
        # green agent (which reads the standard OPENAI_* vars unchanged).
        self._model = (os.getenv("OPENAI_MODEL_PURPLE", "").strip()
                       or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self._api_key = (os.getenv("OPENAI_API_KEY_PURPLE", "").strip()
                         or os.getenv("OPENAI_API_KEY", "").strip())
        self._base_url = (os.getenv("OPENAI_BASE_URL_PURPLE", "").strip()
                          or os.getenv("OPENAI_BASE_URL", "").strip() or None)
        self._client = _make_openai_client(self._api_key, self._base_url)
        logger.info("Purple agent: model=%s base_url=%s", self._model, self._base_url)
        # Azure deployments override the model name
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        if azure_deployment:
            self._model = azure_deployment
        self._config = GridConfig()
        self._planner = BuildPlanner(self._client, self._model, self._config)
        # Conversation history per context for sliding window
        self._history: dict[str, list[dict]] = {}
        self._max_history = 5
        # Pending state: when we ask a question, save everything we need
        # so we can build immediately when the answer arrives.
        self._pending: dict[str, dict] = {}  # ctx_id → {parsed, steps, ...}
        # Ask-once guard: track whether we already asked this round.
        # Cleared on feedback (new round) and successful answer consumption.
        self._asked: set[str] = set()

    # -------------------------------------------------------------------
    # Answer detection helpers
    # -------------------------------------------------------------------
    _ANSWER_RE = re.compile(
        r'^Answer:\s*(.+?)(?:\s*\(.*points.*\))?$',
        re.IGNORECASE,
    )
    _COLOR_NAMES = {"red", "blue", "green", "yellow", "purple", "orange",
                    "white", "black", "brown", "pink", "grey", "gray", "cyan"}

    @classmethod
    def _extract_answer_colors(cls, text: str) -> list[str]:
        """Extract ALL color names from an answer message.

        Returns a list of capitalised colors in the order they appear,
        e.g. ["Blue", "Red"] for "Answer: Blue and Red (-5 points)".
        Returns empty list if no answer pattern found.
        """
        m = cls._ANSWER_RE.match(text.strip())
        if not m:
            return []
        answer_body = m.group(1).strip().rstrip(".,!").lower()
        # Find all color words in order of appearance
        import re as _re
        colors: list[str] = []
        for match in _re.finditer(r'\b(' + '|'.join(cls._COLOR_NAMES) + r')\b', answer_body):
            c = match.group(1).capitalize()
            if c not in colors:  # dedupe, preserve order
                colors.append(c)
        return colors

    _WORD_TO_INT = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }

    @classmethod
    def _extract_answer_count(cls, text: str) -> int | None:
        """Extract a numeric count from an answer message.

        Returns the first integer found in the answer body, or None if
        no answer pattern or no number found.
        E.g. "Answer: 4 blocks (-5 points)" -> 4
             "Answer: three (-5 points for asking)" -> 3
        """
        m = cls._ANSWER_RE.match(text.strip())
        if not m:
            return None
        answer_body = m.group(1).strip().rstrip(".,!").lower()
        # Try digit first
        digit_match = re.search(r'\b(\d+)\b', answer_body)
        if digit_match:
            return int(digit_match.group(1))
        # Try word numbers
        for word, val in cls._WORD_TO_INT.items():
            if re.search(r'\b' + word + r'\b', answer_body):
                return val
        return None

    # -------------------------------------------------------------------
    # Main execute
    # -------------------------------------------------------------------
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        ctx_id = context.context_id or "default"

        if self._debug:
            logger.info("-----")
            logger.info("User input: %s", user_input)

        if not self._api_key:
            await event_queue.enqueue_event(
                new_agent_text_message("[ASK];missing OPENAI_API_KEY", context_id=context.context_id)
            )
            return

        # Parse the green agent message
        parsed = parse_green_message(user_input, self._config)

        if parsed.is_feedback:
            # Store feedback in history for learning
            self._add_to_history(ctx_id, "feedback", parsed.feedback_text)
            logger.info("Received feedback: %s", parsed.feedback_text[:100])
            # Clear any pending state on feedback (new round)
            self._pending.pop(ctx_id, None)
            self._asked.discard(ctx_id)
            await event_queue.enqueue_event(
                new_agent_text_message("[BUILD]", context_id=context.context_id)
            )
            return

        # ------------------------------------------------------------------
        # Check if this is an answer to a question we asked
        # ------------------------------------------------------------------
        pending = self._pending.pop(ctx_id, None)
        ask_type = pending.get("ask_type", "color") if pending else "color"

        # Try to extract the appropriate answer based on what we asked
        answered_colors = self._extract_answer_colors(parsed.instruction_text)
        answered_count = self._extract_answer_count(parsed.instruction_text)

        if pending and (answered_colors or answered_count is not None):
            # Keep ctx_id in self._asked so the re-run cannot ASK again.
            # It will be cleared on the next feedback message (new round).
            self._asked.add(ctx_id)

            if ask_type == "compound":
                # ── Compound answer path: extract both color and count ──
                # Disambiguate color (same logic as color-only path)
                original_instruction = pending["parsed"].instruction_text
                instruction_lower = original_instruction.lower()
                instruction_colors = {
                    c for c in self._COLOR_NAMES if c in instruction_lower
                }

                if len(answered_colors) > 1:
                    new_colors = [
                        c for c in answered_colors
                        if c.lower() not in instruction_colors
                    ]
                    color_str = new_colors[0] if new_colors else answered_colors[-1]
                else:
                    color_str = answered_colors[0] if answered_colors else "Purple"

                logger.info(
                    "Compound answer: color='%s', count=%s",
                    color_str, answered_count,
                )

                # Patch color first, then count
                patched_text = patch_instruction_with_color(
                    pending["parsed"].instruction_text, color_str
                )
                if answered_count is not None:
                    patched_text = patch_instruction_with_count(
                        patched_text, answered_count,
                    )
                    pending["parsed"]._answered_count = answered_count

                logger.info("Compound-patched instruction: %s", patched_text[:200])
                pending["parsed"].instruction_text = patched_text
                response = await self._skills_pipeline(
                    pending["parsed"], ctx_id,
                    pending.get("original_input", ""),
                    override_count=answered_count,
                )
                if response is None:
                    logger.info("Compound-patched pipeline failed, falling back to LLM")
                    count_hint = f" {answered_count}" if answered_count else ""
                    combined = (
                        pending.get("original_input", "")
                        + f"\n\nThe answer to the question is: {color_str}"
                        + (f", {answered_count} blocks" if answered_count else "")
                        + f". Use {color_str} for the unspecified blocks"
                        + (f" and build exactly{count_hint} blocks for the"
                           f" unspecified stack" if answered_count else "")
                        + ". Respond with [BUILD]."
                    )
                    response = await self._direct_llm_call(combined, ctx_id)

            elif ask_type == "count" and answered_count is not None:
                # ── Count answer path ──
                uncounted_color = pending.get("uncounted_color", "")
                logger.info(
                    "Received count answer: %d (color=%s), "
                    "patching instruction and injecting into pipeline",
                    answered_count, uncounted_color or "any",
                )
                # Patch the instruction text to include the answered
                # count BEFORE sending to the LLM planner.  This is
                # critical: the LLM receives "build a stack of 3 blue
                # blocks" instead of "build a blue stack" and produces
                # the correct count in its plan.
                patched_text = patch_instruction_with_count(
                    pending["parsed"].instruction_text,
                    answered_count,
                    target_color=uncounted_color,
                )
                logger.info("Count-patched instruction: %s", patched_text[:200])
                pending["parsed"].instruction_text = patched_text
                pending["parsed"]._answered_count = answered_count
                response = await self._skills_pipeline(
                    pending["parsed"], ctx_id,
                    pending.get("original_input", ""),
                    override_count=answered_count,
                )
                if response is None:
                    logger.info("Count-patched pipeline failed, falling back to LLM")
                    color_hint = f" {uncounted_color}" if uncounted_color else ""
                    combined = (
                        pending.get("original_input", "")
                        + f"\n\n[IMPORTANT: The instruction says to build a"
                        f" stack but does not specify how many{color_hint}"
                        f" blocks. The correct number is {answered_count}."
                        f" Build exactly {answered_count}{color_hint} blocks"
                        f" for the unspecified stack. Do NOT ask questions."
                        f" Respond ONLY with [BUILD].]"
                    )
                    response = await self._direct_llm_call(combined, ctx_id)

            else:
                # ── Color answer path (existing logic) ──
                # Disambiguate multi-color answers.
                # The green agent often answers "Blue and Green" where one
                # color matches what is already in the instruction and the
                # other is the NEW color for the unspecified blocks.
                original_instruction = pending["parsed"].instruction_text
                instruction_lower = original_instruction.lower()
                instruction_colors = {
                    c for c in self._COLOR_NAMES if c in instruction_lower
                }

                if len(answered_colors) > 1:
                    new_colors = [
                        c for c in answered_colors
                        if c.lower() not in instruction_colors
                    ]
                    color_str = new_colors[0] if new_colors else answered_colors[-1]
                    logger.info(
                        "Multi-color answer %s, instruction has %s, using '%s'",
                        answered_colors, instruction_colors, color_str,
                    )
                else:
                    color_str = answered_colors[0] if answered_colors else "Purple"

                logger.info("Received answer color '%s', patching instruction and re-planning", color_str)
                patched_text = patch_instruction_with_color(
                    pending["parsed"].instruction_text, color_str
                )
                pending["parsed"].instruction_text = patched_text
                logger.info("Patched instruction: %s", patched_text[:200])
                response = await self._skills_pipeline(
                    pending["parsed"], ctx_id, pending.get("original_input", "")
                )
                if response is None:
                    logger.info("Patched pipeline failed, falling back to LLM")
                    combined = (
                        pending.get("original_input", "")
                        + f"\n\nThe answer to the question is: {color_str}. "
                        "Use this color for the unspecified blocks. Respond with [BUILD]."
                    )
                    response = await self._direct_llm_call(combined, ctx_id)
        else:
            # Normal instruction ── run the full pipeline
            response = await self._skills_pipeline(parsed, ctx_id, user_input)

            if response is None:
                logger.info("Skills pipeline failed, falling back to direct LLM")
                response = await self._direct_llm_call(user_input, ctx_id)

        # ── HARD GUARD: never send [ASK] more than once per round ──
        # This is the absolute last line of defense.  No matter what
        # logic above produces an [ASK], if we already asked this round
        # we convert it to a direct LLM call and BUILD instead.
        if response.startswith("[ASK]") and ctx_id in self._asked:
            logger.warning(
                "HARD GUARD: suppressing repeated [ASK] → falling back to LLM"
            )
            response = await self._direct_llm_call(user_input, ctx_id)
        elif response.startswith("[ASK]"):
            # First ASK this round — mark it so we never ask again
            self._asked.add(ctx_id)

        # Store in history
        self._add_to_history(ctx_id, "instruction", parsed.instruction_text)
        self._add_to_history(ctx_id, "response", response)

        # ── LAST RESORT: if response is empty [BUILD] and we have start
        # blocks, return the existing structure.  Both score -10, but
        # returning existing blocks is more defensible and may help the
        # green agent's state tracking. ──
        if response == "[BUILD]" and parsed.start_grid and parsed.start_grid.blocks:
            existing = format_build_response(parsed.start_grid)
            logger.warning(
                "Empty [BUILD] replaced with existing %d start blocks "
                "(LLM unavailable, returning start structure as fallback)",
                len(parsed.start_grid.blocks),
            )
            response = existing

        await event_queue.enqueue_event(
            new_agent_text_message(response, context_id=context.context_id)
        )

    async def _skills_pipeline(
        self, parsed: ParsedInstruction, ctx_id: str, original_input: str = "",
        override_count: int | None = None,
    ) -> str | None:
        """Run the skills-based build pipeline.

        Returns a [BUILD] or [ASK] response, or None if the pipeline fails.

        Strategy:
        - Detect underspec BEFORE calling the LLM.
        - If both color and count are missing -> ASK a compound question
          to get both in a single -5 cost round.
        - If only color is missing -> ASK the green agent for the color.
        - When the answer arrives, the answered color is patched directly into
          the instruction text so the LLM receives a fully-specified prompt.
        - If only count is missing but color is known, ASK for the count.
          Heuristic count inference is only 64.6% accurate; asking is +EV.
        - Only one question per round.
        """
        try:
            # ── Step 1: Pre-LLM underspec check on the raw instruction ──
            heuristic_result = detect_underspec_heuristic(parsed.instruction_text)
            logger.info("Heuristic: %s", heuristic_result.details)
            inferred_count = override_count or heuristic_result.inferred_count or 3

            # ── Compound: both color and count missing ──
            if (heuristic_result.has_missing_color
                    and heuristic_result.has_missing_number
                    and ctx_id not in self._asked):
                self._pending[ctx_id] = {
                    "parsed": parsed,
                    "original_input": original_input,
                    "inferred_count": inferred_count,
                    "ask_type": "compound",
                    "uncounted_color": heuristic_result.uncounted_color,
                }
                question = (
                    heuristic_result.suggested_compound_question
                    or "What color should the unspecified blocks be, "
                       "and how many blocks should be in that stack?"
                )
                logger.info("Both color and count missing, compound ask: %s", question)
                return f"[ASK];{question}"

            if heuristic_result.has_missing_color and ctx_id not in self._asked:
                # Color is genuinely missing — ASK before calling the LLM.
                # NOTE: self._asked is set in execute() AFTER the hard guard,
                # not here, so the hard guard doesn't block the first ASK.
                self._pending[ctx_id] = {
                    "parsed": parsed,
                    "original_input": original_input,
                    "inferred_count": inferred_count,
                    "ask_type": "color",
                }
                question = (
                    heuristic_result.suggested_question
                    or "What color should the unspecified blocks be?"
                )
                logger.info("Missing color detected pre-LLM, asking: %s", question)
                return f"[ASK];{question}"

            # Count is missing but color is known — ASK for the count.
            if (heuristic_result.has_missing_number
                    and not heuristic_result.has_missing_color
                    and ctx_id not in self._asked
                    and override_count is None):
                self._pending[ctx_id] = {
                    "parsed": parsed,
                    "original_input": original_input,
                    "inferred_count": inferred_count,
                    "ask_type": "count",
                    "uncounted_color": heuristic_result.uncounted_color,
                }
                question = (
                    heuristic_result.suggested_count_question
                    or "How many blocks should be in the unspecified stack?"
                )
                logger.info("Missing count detected pre-LLM, asking: %s", question)
                return f"[ASK];{question}"

            if heuristic_result.has_missing_color:
                # Already asked this round — fill with inferred color and continue
                fill = heuristic_result.inferred_color or "Purple"
                logger.warning(
                    "Missing color but already asked this round, "
                    "patching instruction with '%s'", fill,
                )
                patched = patch_instruction_with_color(
                    parsed.instruction_text, fill
                )
                parsed.instruction_text = patched

            # ── Step 2: Analyze existing structure for the planner ──
            structure_info = analyze_structure(parsed.start_grid)
            logger.info("Structure analysis: %s", structure_info.describe()[:200])

            # ── Step 3: Decompose instruction into build steps via LLM ──
            # Retry transient errors (Thor timeouts) with backoff.
            steps = None
            for attempt in range(_LLM_MAX_RETRIES + 1):
                t0 = time.monotonic()
                try:
                    steps = await self._planner.decompose(
                        parsed.instruction_text,
                        parsed.start_grid,
                        parsed.speaker,
                        structure_hint=structure_info.describe(),
                    )
                    elapsed = time.monotonic() - t0
                    if steps:
                        logger.info(
                            "Planner succeeded in %.1fs (attempt %d)",
                            elapsed, attempt + 1,
                        )
                        break
                    else:
                        logger.info("Planner returned no steps (attempt %d)", attempt + 1)
                        break  # Empty steps is not transient, don't retry
                except _TRANSIENT_ERRORS as exc:
                    elapsed = time.monotonic() - t0
                    if attempt < _LLM_MAX_RETRIES:
                        backoff = 2 ** attempt * 5
                        logger.warning(
                            "Planner transient error after %.1fs (attempt %d/%d): %s. "
                            "Retrying in %ds...",
                            elapsed, attempt + 1, _LLM_MAX_RETRIES + 1,
                            type(exc).__name__, backoff,
                        )
                        await asyncio.sleep(backoff)
                    else:
                        logger.error(
                            "Planner transient error after %.1fs, all retries exhausted: %s",
                            elapsed, type(exc).__name__,
                        )
                        return None

            if not steps:
                logger.info("Planner returned no steps, falling back")
                return None

            logger.info("Planner produced %d steps", len(steps))
            for i, s in enumerate(steps):
                logger.info("  Step %d: %s %s count=%s pos=%s", i+1, s.action, s.color, s.count, s.position)

            # Step 3b: Patch chain reference coordinates (before auto-fixes
            # so fixes see resolved literal coordinates)
            steps = patch_chain_references(steps, parsed.start_grid)

            # Step 3c: Auto-fix direction errors (deterministic)
            steps = auto_fix_direction(parsed.instruction_text, steps)

            # Step 3c2: Auto-fix each-end cap positions (deterministic)
            steps = auto_fix_each_end_caps(
                parsed.instruction_text, steps, parsed.start_grid
            )

            # Step 3c3: Auto-fix T-shape extend direction (deterministic)
            steps = auto_fix_t_shape_extend(
                parsed.instruction_text, steps, parsed.start_grid
            )

            # Step 3d: Verify plan against instruction
            verification = verify_plan(
                parsed.instruction_text, steps, len(parsed.start_grid.blocks)
            )
            if verification.has_critical:
                logger.info(
                    "Plan verification found critical issues, re-planning: %s",
                    verification.correction_prompt()[:300],
                )
                # Re-plan with correction hints
                steps = await self._planner.decompose(
                    parsed.instruction_text,
                    parsed.start_grid,
                    parsed.speaker,
                    structure_hint=structure_info.describe(),
                    correction_hint=verification.correction_prompt(),
                )
                if not steps:
                    logger.info("Re-plan returned no steps, falling back")
                    return None
                # Apply auto-fixes again after re-plan
                steps = patch_chain_references(steps, parsed.start_grid)
                steps = auto_fix_direction(parsed.instruction_text, steps)
                steps = auto_fix_each_end_caps(
                    parsed.instruction_text, steps, parsed.start_grid
                )
                steps = auto_fix_t_shape_extend(
                    parsed.instruction_text, steps, parsed.start_grid
                )
                logger.info("Re-planned %d steps", len(steps))
                for i, s in enumerate(steps):
                    logger.info("  Re-step %d: %s %s count=%s pos=%s", i+1, s.action, s.color, s.count, s.position)

            # ── Step 4: Safety net — resolve any Uncolored steps ──
            # Since we patch the instruction before the LLM call, Uncolored
            # steps should be rare.  Fill them silently.
            _UNCOLORED = {"uncolored", "unknown", "unspecified", "?"}
            uncolored_steps = [s for s in steps if s.color.lower() in _UNCOLORED]

            if uncolored_steps:
                fill = heuristic_result.inferred_color or "Purple"
                logger.info(
                    "Safety net: resolving %d Uncolored step(s) to '%s'",
                    len(uncolored_steps), fill,
                )
                for s in uncolored_steps:
                    s.color = fill

            # ── Resolve uncounted steps ──
            for step in steps:
                if isinstance(step.count, str) and step.count.lower() in (
                    "uncounted", "unknown", "unspecified", "?"
                ):
                    logger.info("Resolving Uncounted -> %d", inferred_count)
                    step.count = inferred_count

            # ── Log steps after all fixes (pre-execution) ──
            logger.info("Final steps after all fixes (%d total):", len(steps))
            for i, s in enumerate(steps):
                logger.info("  Final step %d: %s %s count=%s pos=%s", i+1, s.action, s.color, s.count, s.position)

            # ── Step 5: Execute steps deterministically ──
            exec_grid = Grid.from_str(parsed.start_grid.to_str(), config=self._config)
            executor = SpatialExecutor(exec_grid)
            executor.execute_plan(steps)

            # ── Step 6: Format response ──
            response = format_build_response(exec_grid)

            # ── Step 7: Validate ──
            is_valid, errors = validate_build_response(response, self._config)
            if not is_valid:
                logger.warning("Validation failed: %s", errors)
                return None

            logger.info("Skills pipeline produced valid response with %d blocks", len(exec_grid.blocks))
            return response

        except ExecutionError as exc:
            logger.warning("Execution error in skills pipeline: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Unexpected error in skills pipeline: %s", exc)
            return None

    async def _direct_llm_call(self, user_input: str, ctx_id: str) -> str:
        """Fallback: direct LLM call with the original system prompt.

        Retries up to _LLM_MAX_RETRIES times on transient errors (timeouts,
        connection resets) with exponential backoff before giving up.
        """
        messages = [
            {"role": "system", "content": _FALLBACK_SYSTEM_PROMPT},
        ]

        # Add sliding window of history for context
        history = self._history.get(ctx_id, [])
        for entry in history[-self._max_history:]:
            if entry["type"] == "instruction":
                messages.append({"role": "user", "content": entry["content"]})
            elif entry["type"] == "response":
                messages.append({"role": "assistant", "content": entry["content"]})
            elif entry["type"] == "feedback":
                messages.append({"role": "user", "content": entry["content"]})

        messages.append({"role": "user", "content": user_input})

        api_kwargs: dict = dict(
            model=self._model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        extra = vllm_extra_body()
        if extra:
            api_kwargs["extra_body"] = extra

        last_exc: Exception | None = None
        for attempt in range(_LLM_MAX_RETRIES + 1):
            try:
                t0 = time.monotonic()
                completion = await self._client.chat.completions.create(**api_kwargs)
                elapsed = time.monotonic() - t0
                content = (completion.choices[0].message.content or "").strip()
                content = strip_think_tags(content)
                logger.info(
                    "Fallback LLM call succeeded in %.1fs (attempt %d)",
                    elapsed, attempt + 1,
                )

                # Basic cleanup
                content = content.rstrip("] \n")
                if content.startswith("[ASK]"):
                    logger.warning(
                        "Fallback LLM tried to [ASK], converting to empty "
                        "[BUILD]: %s", content[:100],
                    )
                    content = "[BUILD]"
                elif not content.startswith("[BUILD]"):
                    logger.warning(
                        "Fallback LLM response doesn't start with [BUILD]: %s",
                        content[:100],
                    )
                    content = "[BUILD]"

                return content

            except _TRANSIENT_ERRORS as exc:
                last_exc = exc
                elapsed = time.monotonic() - t0
                if attempt < _LLM_MAX_RETRIES:
                    backoff = 2 ** attempt * 5  # 5s, 10s
                    logger.warning(
                        "Fallback LLM transient error after %.1fs (attempt %d/%d): %s. "
                        "Retrying in %ds...",
                        elapsed, attempt + 1, _LLM_MAX_RETRIES + 1,
                        type(exc).__name__, backoff,
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        "Fallback LLM transient error after %.1fs (attempt %d/%d): %s. "
                        "All retries exhausted.",
                        elapsed, attempt + 1, _LLM_MAX_RETRIES + 1,
                        type(exc).__name__,
                    )
            except Exception as exc:
                logger.warning("Fallback LLM call failed (non-transient): %s", exc)
                return "[BUILD]"

        logger.error("Fallback LLM all %d attempts failed: %s", _LLM_MAX_RETRIES + 1, last_exc)
        return "[BUILD]"

    def _add_to_history(self, ctx_id: str, entry_type: str, content: str) -> None:
        """Add an entry to the conversation history with sliding window."""
        if ctx_id not in self._history:
            self._history[ctx_id] = []
        self._history[ctx_id].append({"type": entry_type, "content": content})
        # Trim to max history * 3 entries (instruction + response + feedback per round)
        max_entries = self._max_history * 3
        if len(self._history[ctx_id]) > max_entries:
            self._history[ctx_id] = self._history[ctx_id][-max_entries:]

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenAI purple agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9022, help="Port to bind the server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--card-url", default="", help="URL for the agent card")
    args = parser.parse_args()

    debug_env = os.getenv("AGENT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    debug = args.debug or debug_env
    logging.basicConfig(level=logging.INFO if debug else logging.WARNING)

    card_url = args.card_url
    if not card_url:
        if args.host == "0.0.0.0":
            card_host = "127.0.0.1"
        else:
            card_host = args.host
        card_url = f"http://{card_host}:{args.port}"

    card = prepare_agent_card(card_url)
    request_handler = DefaultRequestHandler(
        agent_executor=OpenAIPurpleAgent(debug=debug),
        task_store=InMemoryTaskStore(),
    )

    logger.info(f"Starting OpenAI purple agent on {args.host}:{args.port} with card URL: {card_url}")
    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
