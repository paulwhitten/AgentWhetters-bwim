"""Build Planner: LLM-based instruction decomposition into structured build steps."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI

from .grid import Grid, GridConfig
from .prompt_enricher import get_enrichments

logger = logging.getLogger(__name__)

DECOMPOSITION_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant for a 3D block-building grid game.

GRID COORDINATE SYSTEM:
- 9x5x9 grid in the x,y,z Euclidean space. Origin (0,0,0) is the center.
- The x-axis is horizontal left-right. Left is negative x, right is positive x. The range is [-400, 400] with steps of 100.
- The y-axis is vertical. Ground level y=50. Each stacked block adds y+=100. The range is [50, 450] (up to 5 blocks high).
- The z-axis is horizontal front-back (near to far). Front (near) is positive z, back (far) is negative z. The range is [-400, 400] with steps of 100.

DIRECTIONS (CRITICAL - memorize these):
- "in front of" / "forward" = +z direction (increasing z)
- "behind" / "back" = -z direction (decreasing z)
- "to the right" = +x direction (increasing x)
- "to the left" = -x direction (decreasing x)
- "on top of" / "stack on" = +y direction (increasing y)

CORNERS:
- "bottom left corner" = (-400, 50, 400) [that's -x, +z]
- "bottom right corner" = (400, 50, 400) [that's +x, +z]
- "top left corner" = (-400, 50, -400) [that's -x, -z]
- "top right corner" = (400, 50, -400) [that's +x, -z]

EDGES (9 squares each):
- "left edge" = x=-400, z varies from -400 to 400 (9 positions)
- "right edge" = x=400, z varies from -400 to 400 (9 positions)
- "top edge" / "back edge" = z=-400, x varies from -400 to 400 (9 positions)
- "bottom edge" / "front edge" = z=400, x varies from -400 to 400 (9 positions)

YOUR TASK:
Given a building instruction and the current grid state, decompose the instruction \
into a sequence of atomic build steps. Output valid JSON only.

OUTPUT FORMAT (JSON object):
{
  "steps": [
    {
      "action": "stack" | "place" | "place_relative" | "extend_row" | "place_at_corners",
      "color": "Red" | "Blue" | ... | "Uncolored",
      "count": <integer> | "Uncounted",
      "position": {
        "x": <int>, "z": <int>
      }
      OR
      "position": {
        "relative_to": "origin" | "existing_<Color>_stack_at_<x>_<z>" | "leftmost" | "rightmost" | "frontmost" | "backmost",
        "direction": "right" | "left" | "front" | "behind" | "on_top",
        "distance": <int, default 1>
      }
    }
  ]
}

POSITION AND THE Y-AXIS:
- JSON position specifies x and z ONLY. Y is never included in the output.
- Y is determined automatically by the execution engine:
  * First block at any (x,z) → y=50 (ground level).
  * Each additional block at the same (x,z) → y increments by 100 (150, 250, 350, 450).
  * "stack" with count=N places N blocks vertically from the next available y at that (x,z).
- Think of the x-z plane as a 2D board where blocks stack upward by gravity.
- Use full 3D (x,y,z) coordinates in your spatial reasoning to verify logic, but \
output only {"x":…, "z":…} in the JSON steps.

RULES:
- COLOR HANDLING (important):
  * If a step's color is EXPLICITLY stated in the instruction → use that color.
  * If blocks are mentioned WITHOUT a color but the instruction mentions other colors → \
use "Uncolored". Do NOT guess or infer. We will ask the user for the right color.
  * E.g., "Put a red block in each corner, then stack two blocks on top" → \
the corner blocks are "Red", the top blocks are "Uncolored" (not stated, could be any color).
  * E.g., "Build three blue blocks, then stack blocks to the left" → \
the blue blocks are "Blue", the left stack is "Uncolored".
  * If NO colors are mentioned at all in the entire instruction → use "Uncolored" for all steps.
- Use "Uncounted" ONLY as a last resort. Instead, infer the count:
  * "Stack blocks in front of the green ones" → match the green stack height.
  * "Build a stack to the left" → match the referenced stack height.
  * "Finish with a yellow stack" → match the most recently built stack height.
  * When in doubt, use 3.
- Each step should be ONE atomic action (one stack, one block, one row extension).
- "stack" means place N blocks vertically at one (x,z) position. Y auto-increments from ground or existing blocks.
- "place" means place a single block at an absolute position.
- "place_relative" means place a single block relative to a reference.
- "extend_row" means place N blocks in a line starting FROM a position (inclusive).
  The first block is placed AT the start position, subsequent blocks step in the direction.
  Example: "row of 3 starting at origin going left" → start at (0,50,0), places at (0,50,0), (-100,50,0), (-200,50,0).
- "place_at_corners" means place blocks at grid corners.
- Include the count even for single blocks (count: 1).
- For stacks, count is the number of NEW blocks to add (not total height).
- Order steps so earlier steps don't depend on later ones.
- Output ONLY the JSON object, no explanations or markdown.

IMPORTANT PATTERNS:
- "Along the left edge" = 9 blocks at x=-400 with z from -400 to 400. Use extend_row \
with count=9, position x=-400 z=-400, direction=front.
- "Along the right edge" = 9 blocks at x=400, same z range.
- "Along the top/back edge" = 9 blocks at z=-400 with x from -400 to 400. Use extend_row \
with count=9, position x=-400 z=-400, direction=right.
- "Along the bottom/front edge" = 9 blocks at z=400 with x from -400 to 400.
- "Immediately to the right" = one grid step right (+100 x).
- "Starting at X, going to the left" → extend_row with direction="left".
  Make sure the direction matches the instruction! "going left" = direction left.
- When an instruction says "build a stack of N blocks to the left of existing," the NEW \
stack should be at (existing_x - 100, 50, existing_z). It's ONE step away.
- "Stack N blocks on top of existing" = N new blocks added above the existing ones.
  Stacking is automatic. Just set count=N and position at the existing stack's (x,z).

CRITICAL SPATIAL RULES (many errors come from these):
- "highlighted square" / "middle square" / "highlighted center square" = the origin = (0, 50, 0).
  "Starting from the highlighted square" means the FIRST block goes AT (0, 50, 0).
  "Starting from the middle square and going to the right" = extend_row starting AT x=0, z=0, \
direction=right. NOT at x=100.
- "to the right of the origin" = x=100. "Starting at the square to the right of the origin" \
means first block at (100, 50, 0).
- "each end" / "both ends" of a row = the leftmost AND rightmost blocks of that row.
  After extending a row, recalculate the ends! E.g., if start row is at x=[-100,0,100] and we \
add 2 to the right, extended row = [-100,0,100,200,300]. Ends are x=-100 and x=300.
- "in front of the leftmost block" = find the block with lowest x, then go +z from it.
  E.g., leftmost block at (-100, 50, 0) → "in front of" = (-100, 50, 100).
- DIRECTION DOUBLE-CHECK: Before outputting, verify "going to the left" uses direction "left" \
(decreasing x), "going to the right" uses direction "right" (increasing x).

HORIZONTAL vs VERTICAL — only "on top of" increases y:
- "left/right/front/behind" = HORIZONTAL moves (change x or z, keep y=50 for ground blocks).
- ONLY "on top of" means increasing y. All other directions are HORIZONTAL.
- Example: "red on left of green at (0,50,0), yellow on left of red" → \
red at (-100,50,0), yellow at (-200,50,0). Each step shifts x by -100.

CHAIN REFERENCES — track positions step by step:
- When step A says "stack 3 green blocks behind the existing green block at (0,50,0)" \
the green stack is now at (0, 50, -100).
- Step B says "build a yellow stack to the right of the green one" → \
the reference "the green one" is the stack at (0, 50, -100), so yellow goes at (100, 50, -100).
- WRONG: placing yellow at (100, 50, 0) — that's right of the ORIGINAL block, not the stack.
- Similarly: "behind the rightmost blue block at (100,50,0), build red" → red at (100,50,-100). \
"Build yellow directly to the right of the red one" → yellow at (200,50,-100).
- "To the right of the ROW" = at position (rightmost_x + 100, z), starting at y=50 (ground level). \
Do NOT stack on top of a block IN the row. The new stack goes AFTER the row's last block. \
Row at x=[100,200,300] → "stack to the right of this row" = x=400, y=50. \
WRONG: placing a stack ON TOP of row block at x=200, y=150.

T/L SHAPE EXTENSION:
- For L shapes, the "longer side" is the arm with MORE blocks along it.
- "extend the longer side by 2" = add 2 blocks AWAY FROM THE JUNCTION (corner of the L).
  The junction is where the two arms meet. Extend from the END FARTHEST from the junction.
- An L at positions [(0,50,-100),(0,50,0),(0,50,100),(100,50,100)] has longer arm along z-axis \
(3 blocks: z=-100,0,100 at x=0) and short arm along x-axis (1 block at x=100, z=100). \
Junction at z=100 where the arms meet. Extend longer away from junction: (0,50,-200) \
and (0,50,-300). NOT (0,50,200) — that goes through the junction and breaks the L-shape. \
"Add block to shorter side" = continue outward → (200,50,100).
- NEVER stack on top when told to "extend" — extending is always horizontal.
- T shape: the "base" / "longer base" is the stem (longer arm). To identify: count blocks \
along each axis — whichever axis has MORE is the "longer base" / stem. \
"Arms" are the endpoints of the crossbar. "Add to each arm" = add at the ends of \
the crossbar, extending outward, not on top.

IDENTIFYING EXISTING STRUCTURES (do this BEFORE placing relative to anything):
When the instruction says "relative to", "next to", "behind the row", "to the left of the stack", \
etc., your FIRST reasoning step must IDENTIFY the reference structure from the grid state:

For a single block or stack:
  1. Find the block(s) matching the description (color, position words like "leftmost").
  2. Note its coordinates: (x, y, z). For a stack, note the base (x, z) and height.
  3. Determine the direction word → axis and sign: "right"=+x, "left"=-x, "front"=+z, "behind"=-z, "on top"=+y.
  4. Compute the target: reference coordinate ± 100 on the relevant axis.

For a row:
  1. Find all blocks at the same y that share a common axis value (same x = z-row, same z = x-row).
  2. Determine which axis varies → that is the row's axis and direction.
     E.g., blocks at (-100,50,0), (0,50,0), (100,50,0) → z is fixed at 0, x varies → this is an x-row.
  3. Determine the range: min and max on the varying axis → row spans x=-100 to x=100.
  4. Determine the extension direction from context:
     "extend to the right" → +x from max x. "extend behind" on a z-row → -z from min z.
  5. For "each end": the endpoints are (min, fixed) and (max, fixed) on the varying axis.

Example identification reasoning:
  Grid has: Red at (-200,50,100), (-100,50,100), (0,50,100), (100,50,100).
  → IDENTIFY: 4 blocks share z=100, x varies from -200 to 100 → x-row at z=100, direction +x.
  → Leftmost end: x=-200. Rightmost end: x=100.
  → "extend by 2 to the right" → new blocks at x=200 and x=300, both z=100.
  → "stack on the leftmost" → stack at x=-200, z=100, y auto-stacks above 50.

WORKED EXAMPLES (use these as templates for similar instructions):

EXAMPLE 1 — Chain reference with "in front of"
Instruction: "Stack three green blocks behind the existing green block. Build a yellow stack to the right of the green one."
Grid: Green block at (0,50,0).
Reasoning:
1. "behind" = -z. Existing green at (0,50,0) → new stack at x=0, z=0-100=-100. Three blocks auto-stack at y=50,150,250.
2. "the green one" = stack just built at z=-100 → "to the right" = x=0+100=100, same z=-100.
3. Yellow count not specified → match green height = 3.
→ JSON: positions use x,z only (y auto-stacked): {"x":0,"z":-100} and {"x":100,"z":-100}.
Output:
{"steps":[
  {"action":"stack","color":"Green","count":3,"position":{"x":0,"z":-100}},
  {"action":"stack","color":"Yellow","count":3,"position":{"x":100,"z":-100}}
]}

EXAMPLE 2 — Row going LEFT + stack in front of last block
Instruction: "Build a row of three purple blocks, starting at the origin and going to the left. Stack four blue blocks in front of the last purple block."
Grid: empty.
Reasoning:
1. "going to the left" = direction left (-x). Start at origin x=0, z=0: row at x=0, -100, -200 (all y=50 ground).
2. "last purple block" = last placed at x=-200, z=0.
3. "in front of" = +z → blue stack at x=-200, z=100. Count=4, auto-stacking y=50,150,250,350.
→ JSON: extend_row at {"x":0,"z":0,"direction":"left"}, stack at {"x":-200,"z":100}. Y omitted from JSON.
Output:
{"steps":[
  {"action":"extend_row","color":"Purple","count":3,"position":{"x":0,"z":0,"direction":"left"}},
  {"action":"stack","color":"Blue","count":4,"position":{"x":-200,"z":100}}
]}

EXAMPLE 3 — "In front of each" = horizontal, not stacking
Instruction: "Place one purple block on the highlighted square and one to its right. Place one red block in front of each purple block."
Grid: empty.
Reasoning:
1. Highlighted square = origin → purple at x=0, z=0 (y=50 ground). "to its right" → purple at x=100, z=0.
2. "in front of each purple" = +z for each → red at x=0, z=100 and x=100, z=100. All single blocks at y=50.
→ JSON: four place steps with {"x":0,"z":0}, {"x":100,"z":0}, {"x":0,"z":100}, {"x":100,"z":100}. No y needed.
Output:
{"steps":[
  {"action":"place","color":"Purple","count":1,"position":{"x":0,"z":0}},
  {"action":"place","color":"Purple","count":1,"position":{"x":100,"z":0}},
  {"action":"place","color":"Red","count":1,"position":{"x":0,"z":100}},
  {"action":"place","color":"Red","count":1,"position":{"x":100,"z":100}}
]}

EXAMPLE 4 — Edge placement: left edge = fixed x=-400
Instruction: "Place nine purple blocks along the grid's left edge. Immediately to the right, build a row of nine yellow blocks."
Grid: has some existing blocks.
Reasoning:
1. "left edge" = fixed x=-400, z varies from -400 to 400 (9 positions). All at y=50 ground.
2. "Immediately to the right" = x=-400+100=-300, same z range.
→ JSON: two extend_row steps at {"x":-400,"z":-400,"direction":"front"} and {"x":-300,"z":-400,"direction":"front"}, count=9 each.
Output:
{"steps":[
  {"action":"extend_row","color":"Purple","count":9,"position":{"x":-400,"z":-400,"direction":"front"}},
  {"action":"extend_row","color":"Yellow","count":9,"position":{"x":-300,"z":-400,"direction":"front"}}
]}

EXAMPLE 5 — T shape: "add to each arm" = place at ends of crossbar
Instruction: "Keeping the T shape, extend the existing green structure by adding two green blocks to the longer base. Then add one purple block to each arm."
Grid: T-shape with crossbar at z=-100 (x=-100,0,100) and stem at x=0 going +z (z=0,100,200).
Reasoning:
1. Crossbar: 3 blocks along x at z=-100. Stem: 3 blocks along +z at x=0. Stem = longer base (extends in +z direction).
2. "extend by 2" = continue +z from stem end (z=200) → z=300 and z=400, both at x=0, y=50.
3. "each arm" = tips of crossbar at x=-100 and x=100. "add to each arm" = extend outward → x=-200 and x=200, both z=-100, y=50.
→ JSON: extend_row at {"x":0,"z":300,"direction":"front"} count=2, two place steps at {"x":-200,"z":-100} and {"x":200,"z":-100}.
Output:
{"steps":[
  {"action":"extend_row","color":"Green","count":2,"position":{"x":0,"z":300,"direction":"front"}},
  {"action":"place","color":"Purple","count":1,"position":{"x":-200,"z":-100}},
  {"action":"place","color":"Purple","count":1,"position":{"x":200,"z":-100}}
]}

EXAMPLE 6 — Extend existing row + "block on top of each end"
Instruction: "There is a red row. Extend it by adding two red blocks to its right. Place one block on top of each end."
Grid: Red blocks at (-100,50,0) and (0,50,0).
Reasoning:
1. Rightmost red at x=0, z=0. "add two to its right" → extend at x=100 and x=200, both z=0, y=50 ground.
2. Extended row spans x=-100 to x=200. Ends: x=-100 and x=200.
3. "on top of each end" = stack 1 block on each. At x=-100: existing block at y=50 → new at y=150. Same at x=200. Color not stated → "Uncolored".
→ JSON: extend_row at {"x":100,"z":0,"direction":"right"} count=2, two stacks at {"x":-100,"z":0} and {"x":200,"z":0} count=1 each. Y auto-stacks above existing.
Output:
{"steps":[
  {"action":"extend_row","color":"Red","count":2,"position":{"x":100,"z":0,"direction":"right"}},
  {"action":"stack","color":"Uncolored","count":1,"position":{"x":-100,"z":0}},
  {"action":"stack","color":"Uncolored","count":1,"position":{"x":200,"z":0}}
]}

EXAMPLE 7 — "Stack blocks to the right of the green ones" = separate column, same z
Instruction: "Stack four green blocks to the right of the highlighted middle square. Stack blue blocks to the right of the green ones."
Grid: empty.
Reasoning:
1. "to the right of the highlighted middle square" → x=0+100=100, z=0. Stack 4 green (y=50,150,250,350 auto).
2. "to the right of the green ones" → x=100+100=200, z=0. Count not stated → match green height = 4.
→ JSON: two stacks at {"x":100,"z":0} count=4 and {"x":200,"z":0} count=4. Y auto-stacked.
Output:
{"steps":[
  {"action":"stack","color":"Green","count":4,"position":{"x":100,"z":0}},
  {"action":"stack","color":"Blue","count":4,"position":{"x":200,"z":0}}
]}

EXAMPLE 8 — Yellow row going LEFT (not right!)
Instruction: "Starting from the highlighted square, place a horizontal row of three yellow blocks going towards the left side of the grid. Stack two blocks on top of the second yellow block."
Grid: empty.
Reasoning:
1. "going towards the left" = direction left (-x). Start at origin x=0, z=0: row at x=0, -100, -200 (all y=50 ground).
2. "second yellow block" = second placed = x=-100, z=0.
3. Stack 2 on top at x=-100, z=0 → new blocks at y=150,250 (above existing y=50). Color not stated → "Uncolored".
→ JSON: extend_row at {"x":0,"z":0,"direction":"left"}, stack at {"x":-100,"z":0} count=2. Y auto-stacks above existing.
Output:
{"steps":[
  {"action":"extend_row","color":"Yellow","count":3,"position":{"x":0,"z":0,"direction":"left"}},
  {"action":"stack","color":"Uncolored","count":2,"position":{"x":-100,"z":0}}
]}

EXAMPLE 9 — Identify existing row, extend it, place relative to endpoint
Instruction: "Extend the existing blue row by two blocks. Then stack two green blocks behind the leftmost blue block."
Grid: Blue blocks at (-100,50,0), (0,50,0), (100,50,0).
Reasoning:
1. IDENTIFY the row: 3 blue blocks share z=0, x varies -100→0→100 → this is an x-row at z=0, extending in +x direction. Leftmost: x=-100. Rightmost: x=100.
2. "extend by two" — row runs in +x, continue: x=100+100=200 and x=200+100=300. Two new blue blocks at z=0, y=50 ground.
3. IDENTIFY for relative placement: "leftmost blue block" = x=-100, z=0 (unchanged by the rightward extension).
4. "behind the leftmost" = -z direction → z=0-100=-100, x=-100. Stack 2 green, y=50,150 auto.
→ JSON: extend_row at {"x":200,"z":0,"direction":"right"} count=2, stack at {"x":-100,"z":-100} count=2.
Output:
{"steps":[
  {"action":"extend_row","color":"Blue","count":2,"position":{"x":200,"z":0,"direction":"right"}},
  {"action":"stack","color":"Green","count":2,"position":{"x":-100,"z":-100}}
]}
"""


@dataclass
class BuildStep:
    """A single atomic build step."""
    action: str  # "stack", "place", "place_relative", "extend_row", "place_at_corners"
    color: str  # "Red", "Uncolored", etc.
    count: int | str  # integer or "Uncounted"
    position: dict = field(default_factory=dict)  # absolute or relative position

    @classmethod
    def from_dict(cls, d: dict) -> BuildStep:
        return cls(
            action=d.get("action", "place"),
            color=d.get("color", "Uncolored"),
            count=d.get("count", 1),
            position=d.get("position", {}),
        )


class BuildPlanner:
    """Decomposes building instructions into structured steps using an LLM."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        config: GridConfig | None = None,
    ):
        self._client = client
        self._model = model
        self._config = config or GridConfig()

    async def decompose(
        self,
        instruction: str,
        start_grid: Grid,
        speaker: str = "",
        structure_hint: str = "",
        correction_hint: str = "",
    ) -> list[BuildStep]:
        """Decompose an instruction into build steps.

        Returns a list of BuildStep objects. If decomposition fails,
        returns an empty list (caller should fall back to direct LLM call).

        Parameters
        ----------
        structure_hint : str
            Pre-computed analysis of the start grid's shape/geometry.
        correction_hint : str
            Feedback from the plan verifier about errors in a previous plan.
        """
        grid_description = start_grid.describe()

        user_prompt = ""
        if structure_hint:
            user_prompt += f"{structure_hint}\n\n"
        user_prompt += (
            f"CURRENT GRID STATE:\n{grid_description}\n\n"
            f"INSTRUCTION: {instruction}\n"
        )
        if speaker:
            user_prompt = f"SPEAKER: {speaker}\n" + user_prompt
        if correction_hint:
            user_prompt += f"\n{correction_hint}\n"

        # Adaptive enrichment: inject concept reminders for difficult phrases
        enrichment = get_enrichments(instruction)
        if enrichment:
            user_prompt += f"\n{enrichment}\n"
            logger.info("Enrichment injected for: %s", instruction[:80])

        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )

            content = (completion.choices[0].message.content or "").strip()
            logger.info("Build planner raw output: %s", content[:500])

            return self._parse_response(content)

        except Exception as exc:
            logger.warning("Build planner failed: %s", exc)
            return []

    def _parse_response(self, content: str) -> list[BuildStep]:
        """Parse LLM JSON response into BuildStep objects."""
        try:
            # Strip any markdown code fences
            content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE)

            data = json.loads(content)

            if isinstance(data, dict) and "steps" in data:
                steps_raw = data["steps"]
            elif isinstance(data, list):
                steps_raw = data
            else:
                logger.warning("Unexpected planner response structure: %s", type(data))
                return []

            steps = []
            for step_dict in steps_raw:
                if isinstance(step_dict, dict):
                    steps.append(BuildStep.from_dict(step_dict))

            return steps

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse planner response: %s", exc)
            return []
