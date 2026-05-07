"""
IGLU 2.5-D evaluation harness.

Loads the IGLU single-turn dataset, sends each instruction through
either a bare LLM or the 2.5-D pipeline, and scores the result using
the IGLU maximal-intersection F1 metric.

Usage:
    python iglu/evaluate.py --mode bare --limit 50
    python iglu/evaluate.py --mode pipeline --limit 50
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# IGLU constants
# ---------------------------------------------------------------------------

BUILD_ZONE_SIZE = (9, 11, 11)  # Y, X, Z
GROUND_LEVEL = 63
COORD_SHIFT_X = 5
COORD_SHIFT_Z = 5

# voxelworld colour id -> IGLU colour id
BLOCK_MAP = {
    0: 0,    # air
    57: 1,   # blue
    50: 6,   # yellow
    59: 2,   # green
    47: 4,   # orange
    56: 5,   # purple
    60: 3,   # red
    86: 1,   # blue (freeze)
    87: 6,   # yellow (freeze)
    88: 2,   # green (freeze)
    89: 4,   # orange (freeze)
    90: 5,   # purple (freeze)
    91: 3,   # red (freeze)
}

IGLU_COLOR_NAMES = {
    1: "blue", 2: "green", 3: "red",
    4: "orange", 5: "purple", 6: "yellow",
}

COLOR_NAME_TO_IGLU = {v: k for k, v in IGLU_COLOR_NAMES.items()}


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def blocks_to_dense(blocks):
    """Convert list of [x, y, z, color_id] to dense (9, 11, 11) grid."""
    grid = np.zeros(BUILD_ZONE_SIZE, dtype=np.int32)
    for x, y, z, cid in blocks:
        iglu_cid = BLOCK_MAP.get(cid, 0)
        if iglu_cid == 0:
            continue
        gy = y - GROUND_LEVEL + 1
        gx = x + COORD_SHIFT_X
        gz = z + COORD_SHIFT_Z
        if 0 <= gy < 9 and 0 <= gx < 11 and 0 <= gz < 11:
            grid[gy, gx, gz] = iglu_cid
    return grid


def dense_to_block_list(grid):
    """Convert dense grid back to human-readable block list."""
    blocks = []
    for gy in range(9):
        for gx in range(11):
            for gz in range(11):
                cid = grid[gy, gx, gz]
                if cid != 0:
                    x = gx - COORD_SHIFT_X
                    y = gy + GROUND_LEVEL - 1
                    z = gz - COORD_SHIFT_Z
                    blocks.append((x, y, z, IGLU_COLOR_NAMES.get(cid, "unknown")))
    return blocks


def grid_to_text(grid):
    """Convert dense grid to human-readable text for prompts."""
    blocks = dense_to_block_list(grid)
    if not blocks:
        return "The build zone is empty."
    lines = []
    for x, y, z, color in blocks:
        lines.append(f"  {color} block at x={x}, y={y}, z={z}")
    return "Current blocks in the build zone:\n" + "\n".join(lines)


def grid_to_text_2d(grid):
    """Convert dense grid to text showing only (color, x, z) with stack counts.

    For the 2.5-D mode, we describe existing blocks by column occupancy
    so the LLM understands stacking without seeing y-coordinates.
    """
    blocks = dense_to_block_list(grid)
    if not blocks:
        return "The build zone is empty."
    # Group by (x, z) column, list bottom-to-top
    columns = {}
    for x, y, z, color in sorted(blocks, key=lambda b: (b[0], b[2], b[1])):
        columns.setdefault((x, z), []).append(color)
    lines = []
    for (x, z), colors in sorted(columns.items()):
        if len(colors) == 1:
            lines.append(f"  {colors[0]} block at x={x}, z={z}")
        else:
            stack_desc = ", ".join(colors)
            lines.append(f"  stack at x={x}, z={z} (bottom to top): {stack_desc}")
    return "Current blocks in the build zone:\n" + "\n".join(lines)


def maximal_intersection(grid, target_grid):
    """Compute maximal intersection over translations and rotations."""
    Y, X, Z = BUILD_ZONE_SIZE
    target_size = int((target_grid != 0).sum())
    if target_size == 0:
        return 0, target_size

    max_int = 0
    current_target = target_grid.copy()

    for rot in range(4):
        for dx in range(-X + 1, X):
            for dz in range(-Z + 1, Z):
                x_t = slice(max(dx, 0), X + min(dx, 0))
                z_t = slice(max(dz, 0), Z + min(dz, 0))
                x_g = slice(max(-dx, 0), X + min(-dx, 0))
                z_g = slice(max(-dz, 0), Z + min(-dz, 0))

                sls_target = current_target[:, x_t, z_t]
                sls_grid = grid[:, x_g, z_g]

                intersection = int(
                    ((sls_target == sls_grid) & (sls_target != 0)).sum()
                )
                max_int = max(max_int, intersection)

        # Rotate 90 degrees around Y axis
        new_target = np.zeros_like(current_target)
        for gx in range(X):
            for gz in range(Z):
                new_target[:, gz, X - gx - 1] = current_target[:, gx, gz]
        current_target = new_target

    return max_int, target_size


def f1_score(grid, target_grid):
    """Compute F1 between predicted grid and target grid."""
    intersection, target_size = maximal_intersection(grid, target_grid)
    pred_size = int((grid != 0).sum())

    if pred_size == 0 and target_size == 0:
        return 1.0, 1.0, 1.0

    precision = intersection / pred_size if pred_size > 0 else 0.0
    recall = intersection / target_size if target_size > 0 else 0.0

    if precision + recall == 0:
        return 0.0, precision, recall

    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_singleturn_dataset(data_dir, limit=None):
    """Load IGLU single-turn tasks from disk.

    Returns list of dicts with keys:
        game_id, instruction, initial_blocks, target_blocks,
        initial_grid, target_grid, is_clear
    """
    csv_path = os.path.join(data_dir, "clarifying_questions_train.csv")
    tasks = []
    seen = set()

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_id = row["GameId"]
            if game_id in seen:
                continue
            seen.add(game_id)

            instruction = row["InputInstruction"]
            is_clear = row["IsInstructionClear"] == "Yes"
            init_path = row["InitializedWorldPath"]

            # Load initial world state
            init_full = os.path.join(data_dir, init_path)
            if not os.path.exists(init_full):
                continue
            with open(init_full) as jf:
                init_data = json.load(jf)
            init_blocks = init_data.get("worldEndingState", {}).get("blocks", [])

            # Load target world state
            # Target path uses actionHit subdirectory
            # game_id is like CQ-game-123, target uses game-123
            orig_game_id = game_id[len("CQ-"):] if game_id.startswith("CQ-") else game_id
            target_dir = os.path.join(
                data_dir, "target_world_states", "builder-data", "actionHit",
                orig_game_id
            )
            if not os.path.isdir(target_dir):
                # Try the CQ- prefixed version in another location
                target_dir2 = os.path.join(
                    data_dir, "target_world_states", "builder-data",
                    game_id.lower()
                )
                if os.path.isdir(target_dir2):
                    target_dir = target_dir2
                else:
                    continue

            # Find the step file (usually game-XX-step-action)
            target_files = os.listdir(target_dir)
            if not target_files:
                continue
            target_file = os.path.join(target_dir, target_files[0])
            if os.path.isdir(target_file):
                continue
            with open(target_file) as jf:
                target_data = json.load(jf)
            target_blocks = target_data.get("worldEndingState", {}).get("blocks", [])

            init_grid = blocks_to_dense(init_blocks)
            target_grid = blocks_to_dense(target_blocks)

            # Skip if target is same as initial (nothing to build)
            if np.array_equal(init_grid, target_grid):
                continue

            tasks.append({
                "game_id": game_id,
                "instruction": instruction,
                "initial_blocks": init_blocks,
                "target_blocks": target_blocks,
                "initial_grid": init_grid,
                "target_grid": target_grid,
                "is_clear": is_clear,
            })

            if limit and len(tasks) >= limit:
                break

    return tasks


# ---------------------------------------------------------------------------
# LLM callers
# ---------------------------------------------------------------------------

BARE_SYSTEM_PROMPT = """You are a block-building agent in a Minecraft-like voxel world.
The build zone is an 11x11 grid (x from -5 to 5, z from -5 to 5).
The ground level is y=63. Each block stacked above adds 1 to y (64, 65, etc).
Available colors: blue, green, red, orange, purple, yellow.

Given the current state of the build zone and a building instruction,
output the COMPLETE list of all blocks that should exist after following
the instruction (including any blocks that were already there).

Output format (one block per line, no extra text):
color,x,y,z
color,x,y,z
...

Example:
blue,0,63,0
red,0,64,0
green,1,63,0
"""

PIPELINE_SYSTEM_PROMPT = """You are a block-building planner in a Minecraft-like voxel world.
The build zone is an 11x11 grid (x from -5 to 5, z from -5 to 5).
Available colors: blue, green, red, orange, purple, yellow.

Given the current blocks and a building instruction, output a 2D build plan.
You do NOT need to specify y-coordinates. The system will compute vertical
placement automatically using gravity (blocks stack on top of existing blocks).

Output format: a JSON array of actions. Each action is an object with:
  - "action": one of "place", "stack", "row"
  - "color": block color name
  - "x": x-coordinate (-5 to 5)
  - "z": z-coordinate (-5 to 5)
  - "count": number of blocks (for stack/row)
  - "direction": for row, one of "+x", "-x", "+z", "-z"

Example: place a blue block at origin and stack 3 red blocks there:
[
  {"action": "place", "color": "blue", "x": 0, "z": 0},
  {"action": "stack", "color": "red", "x": 0, "z": 0, "count": 3}
]

IMPORTANT: Only output the JSON array with no other text.
"""

# ---------------------------------------------------------------------------
# 2.5-D only mode: minimal decomposition without BWIM-specific actions
# ---------------------------------------------------------------------------

DECOMP25D_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant for a 3D block-building task on a Minecraft-like voxel grid.

GRID COORDINATE SYSTEM:
- 11x11 grid in the x-z horizontal plane. Origin (0,0) is the center.
- x ranges from -5 to 5 (left to right). z ranges from -5 to 5.
- y is vertical (height). Ground level is y=63. Each stacked block adds y+=1.
- Available colors: blue, green, red, orange, purple, yellow.

IMPORTANT - Y-AXIS HANDLING:
- You do NOT output y-coordinates. The execution engine computes y via gravity.
- "stack" with count=N places N NEW blocks vertically at one (x,z), auto-incrementing y from the top of any existing blocks at that position.
- Think of the x-z plane as a 2D board where blocks stack upward by gravity.
- The existing blocks shown in the grid state are already placed. You only output NEW actions.

YOUR TASK:
Given the current grid state and a building instruction, output ONLY the new build actions needed to fulfill the instruction. Do NOT re-emit existing blocks. The execution engine already has the existing state and will apply your actions on top of it.

Output ONLY valid JSON, no explanations or markdown.

OUTPUT FORMAT (strict JSON):
{
  "steps": [
    {
      "action": "stack" | "place" | "extend_row" | "remove",
      "color": "blue" | "green" | "red" | "orange" | "purple" | "yellow",
      "count": <integer>,
      "position": {"x": <int>, "z": <int>},
      "direction": "x+" | "x-" | "z+" | "z-" (only for extend_row)
    }
  ]
}

ACTIONS:
- "stack": Place count NEW blocks vertically at position (x,z). Y auto-increments from the top of existing blocks at that position.
- "place": Place exactly 1 NEW block at position (x,z). Same as stack with count=1.
- "extend_row": Place count NEW blocks in a horizontal line starting at position, stepping in direction.
  Direction "x+" means increasing x, "x-" means decreasing x, "z+" means increasing z, "z-" means decreasing z.
- "remove": Remove count blocks from the top of the column at position (x,z). Removes from the top down.

RULES:
- Output ONLY new actions to apply. Do NOT include existing blocks.
- For stacks, count is the number of NEW blocks to add (not total height).
- Use lowercase color names.
- count must always be a positive integer.
- If the instruction is empty or no changes are needed, output {"steps": []}.
- Output ONLY the JSON object. No text before or after.

EXAMPLE:
Grid state: "stack at x=0, z=0 (bottom to top): red, red"
Instruction: "Add one more red block on top and place a blue block to its right"
{
  "steps": [
    {"action": "stack", "color": "red", "count": 1, "position": {"x": 0, "z": 0}},
    {"action": "place", "color": "blue", "count": 1, "position": {"x": 1, "z": 0}}
  ]
}
"""


def parse_bare_response(response_text, initial_grid):
    """Parse bare LLM response into a dense grid."""
    grid = np.zeros(BUILD_ZONE_SIZE, dtype=np.int32)
    for line in response_text.strip().split("\n"):
        line = line.strip().strip(",").strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            color_name = parts[0].lower()
            x = int(parts[1])
            y = int(parts[2])
            z = int(parts[3])
            cid = COLOR_NAME_TO_IGLU.get(color_name, 0)
            if cid == 0:
                continue
            gy = y - GROUND_LEVEL + 1
            gx = x + COORD_SHIFT_X
            gz = z + COORD_SHIFT_Z
            if 0 <= gy < 9 and 0 <= gx < 11 and 0 <= gz < 11:
                grid[gy, gx, gz] = cid
        except (ValueError, IndexError):
            continue
    return grid


def parse_decomp25d_response(response_text, initial_grid):
    """Parse 2.5-D JSON plan and execute with gravity placement.

    The LLM outputs only NEW actions (not existing blocks).
    We start from initial_grid and apply actions on top, like the BWIM SpatialExecutor.
    """
    grid = initial_grid.copy()

    try:
        text = response_text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        plan = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to extract JSON from free-form text
        import re as _re
        m = _re.search(r'\{.*\}', text, _re.DOTALL)
        if m:
            try:
                plan = json.loads(m.group())
            except json.JSONDecodeError:
                return grid
        else:
            return grid

    steps = plan.get("steps", []) if isinstance(plan, dict) else []

    for step in steps:
        if not isinstance(step, dict):
            continue
        action = step.get("action", "place")
        color_name = step.get("color", "").lower()
        count = step.get("count", 1)
        pos = step.get("position", {})
        direction = step.get("direction", "x+")

        cid = COLOR_NAME_TO_IGLU.get(color_name, 0)
        if cid == 0:
            continue

        try:
            x = int(pos.get("x", 0))
            z = int(pos.get("z", 0))
            count = int(count) if count else 1
        except (ValueError, TypeError):
            continue

        if action in ("stack", "place"):
            for _ in range(count):
                _place_block(grid, x, z, cid)
        elif action == "remove":
            # Remove blocks from the top of the column at (x, z)
            gx = x + COORD_SHIFT_X
            gz = z + COORD_SHIFT_Z
            if 0 <= gx < 11 and 0 <= gz < 11:
                for _ in range(count):
                    # Find highest block and remove it
                    for gy in range(8, 0, -1):
                        if grid[gy, gx, gz] != 0:
                            grid[gy, gx, gz] = 0
                            break
        elif action == "extend_row":
            # Step in the given direction
            dx, dz = 0, 0
            if direction == "x+":
                dx = 1
            elif direction == "x-":
                dx = -1
            elif direction == "z+":
                dz = 1
            elif direction == "z-":
                dz = -1
            for i in range(count):
                _place_block(grid, x + i * dx, z + i * dz, cid)

    return grid


def execute_pipeline_plan(plan_json, initial_grid):
    """Execute a 2D plan with deterministic vertical placement."""
    grid = initial_grid.copy()

    try:
        if isinstance(plan_json, str):
            # Strip markdown code fences if present
            text = plan_json.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            plan = json.loads(text)
        else:
            plan = plan_json
    except json.JSONDecodeError:
        return grid

    if not isinstance(plan, list):
        return grid

    for action in plan:
        if not isinstance(action, dict):
            continue

        act_type = action.get("action", "place")
        color_name = action.get("color", "").lower()
        cid = COLOR_NAME_TO_IGLU.get(color_name, 0)
        if cid == 0:
            continue

        x = action.get("x", 0)
        z = action.get("z", 0)
        count = action.get("count", 1)
        direction = action.get("direction", "+x")

        if act_type == "place":
            _place_block(grid, x, z, cid)

        elif act_type == "stack":
            for _ in range(count):
                _place_block(grid, x, z, cid)

        elif act_type == "row":
            dx, dz = _direction_to_delta(direction)
            cx, cz = x, z
            for _ in range(count):
                _place_block(grid, cx, cz, cid)
                cx += dx
                cz += dz

    return grid


def _place_block(grid, x, z, cid):
    """Place one block at (x, z) using gravity (2.5-D)."""
    gx = x + COORD_SHIFT_X
    gz = z + COORD_SHIFT_Z
    if not (0 <= gx < 11 and 0 <= gz < 11):
        return
    # Find first empty y starting at ground level (gy=1, which is y=63)
    for gy in range(1, 9):
        if grid[gy, gx, gz] == 0:
            grid[gy, gx, gz] = cid
            return


def _direction_to_delta(direction):
    return {
        "+x": (1, 0), "-x": (-1, 0),
        "+z": (0, 1), "-z": (0, -1),
    }.get(direction, (1, 0))


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(tasks, mode, model="gpt-4o-mini"):
    """Run evaluation on a list of tasks."""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    results = []
    total_f1 = 0.0

    for i, task in enumerate(tasks):
        instruction = task["instruction"]
        initial_grid = task["initial_grid"]
        target_grid = task["target_grid"]

        state_text = grid_to_text(initial_grid)

        if mode == "bare":
            system = BARE_SYSTEM_PROMPT
            user_msg = f"{state_text}\n\nInstruction: {instruction}"
        elif mode == "decomp25d":
            system = DECOMP25D_SYSTEM_PROMPT
            state_text_2d = grid_to_text_2d(initial_grid)
            user_msg = f"{state_text_2d}\n\nInstruction: {instruction}"
        else:
            system = PIPELINE_SYSTEM_PROMPT
            user_msg = f"{state_text}\n\nInstruction: {instruction}"

        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            elapsed = time.time() - t0
            response_text = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [{i+1}] LLM error: {e}")
            results.append({"game_id": task["game_id"], "f1": 0.0, "error": str(e)})
            continue

        if mode == "bare":
            pred_grid = parse_bare_response(response_text, initial_grid)
        elif mode == "decomp25d":
            pred_grid = parse_decomp25d_response(response_text, initial_grid)
        else:
            pred_grid = execute_pipeline_plan(response_text, initial_grid)

        f1, prec, rec = f1_score(pred_grid, target_grid)
        total_f1 += f1

        results.append({
            "game_id": task["game_id"],
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "elapsed": elapsed,
        })

        status = "OK" if f1 > 0.5 else "LOW"
        print(f"  [{i+1}/{len(tasks)}] {task['game_id']}: "
              f"F1={f1:.3f} P={prec:.3f} R={rec:.3f} ({elapsed:.1f}s) [{status}]")

    mean_f1 = total_f1 / len(tasks) if tasks else 0.0
    return results, mean_f1


def main():
    parser = argparse.ArgumentParser(description="IGLU 2.5-D evaluation")
    parser.add_argument("--mode", choices=["bare", "pipeline", "decomp25d"],
                        default="decomp25d",
                        help="bare = direct LLM with y, pipeline = structured JSON actions, "
                             "decomp25d = 2.5-D only (color,x,z with gravity)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model name")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max tasks to evaluate")
    parser.add_argument("--clear-only", action="store_true",
                        help="Only use tasks marked as clear instructions")
    parser.add_argument("--additive-only", action="store_true",
                        help="Exclude destroy/remove instructions")
    parser.add_argument("--gravity-only", action="store_true",
                        help="Only use tasks where target has no floating blocks")
    parser.add_argument("--data-dir", default=None,
                        help="Path to singleturn dataset directory")
    args = parser.parse_args()

    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Default: look relative to this script
        data_dir = os.path.join(
            os.path.dirname(__file__),
            "iglu-datasets", "datasets", "singleturn"
        )

    print(f"Loading IGLU single-turn dataset from {data_dir}")
    tasks = load_singleturn_dataset(data_dir, limit=None)
    print(f"Loaded {len(tasks)} tasks")

    if args.clear_only:
        tasks = [t for t in tasks if t["is_clear"]]
        print(f"Filtered to {len(tasks)} clear-instruction tasks")

    if args.additive_only:
        tasks = [t for t in tasks if
                 "destroy" not in t["instruction"].lower() and
                 "remove" not in t["instruction"].lower()]
        print(f"Filtered to {len(tasks)} additive-only tasks")

    if args.gravity_only:
        def _has_floating(grid):
            for gy in range(2, 9):
                for gx in range(11):
                    for gz in range(11):
                        if grid[gy, gx, gz] != 0 and grid[gy-1, gx, gz] == 0:
                            return True
            return False
        tasks = [t for t in tasks if not _has_floating(t["target_grid"])]
        print(f"Filtered to {len(tasks)} gravity-compatible tasks")

    if args.limit and args.limit < len(tasks):
        tasks = tasks[:args.limit]
        print(f"Limited to {args.limit} tasks")

    print(f"\nRunning {args.mode} evaluation with {args.model}")
    print("=" * 60)

    results, mean_f1 = run_evaluation(tasks, args.mode, args.model)

    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Tasks: {len(results)}")
    print(f"Mean F1: {mean_f1:.4f}")

    # Save results
    filter_suffix = ""
    if args.gravity_only:
        filter_suffix += "_gravity"
    if args.additive_only:
        filter_suffix += "_additive"
    out_path = os.path.join(
        os.path.dirname(__file__),
        f"results_{args.mode}_{args.model.replace('/', '_')}_{len(results)}{filter_suffix}.json"
    )
    with open(out_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "model": args.model,
            "n_tasks": len(results),
            "mean_f1": mean_f1,
            "results": results,
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
