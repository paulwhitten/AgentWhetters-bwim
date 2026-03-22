"""Tests for the skills package: grid, spatial, instruction parser, executor, formatter."""

import csv
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pragmatic_builder"))

from skills.grid import Block, Grid, GridConfig, direction_offset, corner_position
from skills.spatial import is_touching, relationship, is_connected, blocks_next_to
from skills.instruction_parser import parse_green_message
from skills.spatial_executor import SpatialExecutor, ExecutionError
from skills.build_planner import BuildStep
from skills.response_formatter import format_build_response, validate_build_response
from skills.underspec_detector import (
    detect_underspec_heuristic,
    detect_underspec_from_plan,
    patch_instruction_with_color,
)

import pytest


# --- Grid tests ---

class TestBlock:
    def test_from_str(self):
        b = Block.from_str("Red,0,50,0")
        assert b.color == "Red"
        assert b.x == 0
        assert b.y == 50
        assert b.z == 0

    def test_str_round_trip(self):
        b = Block(color="Blue", x=100, y=150, z=-200)
        assert str(b) == "Blue,100,150,-200"
        b2 = Block.from_str(str(b))
        assert b == b2

    def test_moved(self):
        b = Block(color="Red", x=0, y=50, z=0)
        b2 = b.moved(dx=100, dz=100)
        assert b2.x == 100
        assert b2.z == 100
        assert b2.y == 50


class TestGrid:
    def test_from_str_empty(self):
        g = Grid.from_str("")
        assert len(g.blocks) == 0

    def test_from_str_single(self):
        g = Grid.from_str("Red,0,50,0")
        assert len(g.blocks) == 1
        assert g.blocks[0].color == "Red"

    def test_from_str_multiple(self):
        g = Grid.from_str("Red,0,50,0;Blue,100,50,0")
        assert len(g.blocks) == 2

    def test_next_y_empty(self):
        g = Grid()
        assert g.next_y(0, 0) == 50

    def test_next_y_with_block(self):
        g = Grid.from_str("Red,0,50,0")
        assert g.next_y(0, 0) == 150

    def test_stack_height(self):
        g = Grid.from_str("Red,0,50,0;Red,0,150,0;Red,0,250,0")
        assert g.stack_height(0, 0) == 3
        assert g.stack_height(100, 0) == 0

    def test_to_build_response(self):
        g = Grid.from_str("Red,0,50,0")
        resp = g.to_build_response()
        assert resp.startswith("[BUILD]")
        assert "Red,0,50,0" in resp

    def test_describe(self):
        g = Grid.from_str("Red,0,50,0;Red,0,150,0")
        desc = g.describe()
        assert "2 block(s)" in desc
        assert "Red" in desc


class TestGridConfig:
    def test_valid_xz(self):
        cfg = GridConfig()
        assert -400 in cfg.valid_xz
        assert 400 in cfg.valid_xz
        assert 0 in cfg.valid_xz
        assert len(cfg.valid_xz) == 9

    def test_valid_y(self):
        cfg = GridConfig()
        assert cfg.valid_y == [50, 150, 250, 350, 450]

    def test_is_valid_position(self):
        cfg = GridConfig()
        assert cfg.is_valid_position(0, 50, 0)
        assert cfg.is_valid_position(-400, 450, 400)
        assert not cfg.is_valid_position(50, 50, 0)  # 50 not a valid x
        assert not cfg.is_valid_position(0, 100, 0)  # 100 not a valid y

    def test_corner_positions(self):
        cfg = GridConfig()
        corners = cfg.corner_positions
        assert corners["top_left"] == (-400, -400)
        assert corners["bottom_right"] == (400, 400)


class TestDirectionOffset:
    def test_right(self):
        dx, dz = direction_offset("right")
        assert dx == 100
        assert dz == 0

    def test_left(self):
        dx, dz = direction_offset("left")
        assert dx == -100
        assert dz == 0

    def test_front(self):
        dx, dz = direction_offset("front")
        assert dx == 0
        assert dz == 100

    def test_behind(self):
        dx, dz = direction_offset("behind")
        assert dx == 0
        assert dz == -100

    def test_in_front_of(self):
        dx, dz = direction_offset("in front of")
        assert dx == 0
        assert dz == 100


# --- Spatial tests ---

class TestSpatial:
    def test_is_touching_adjacent(self):
        cfg = GridConfig()
        a = Block(color="Red", x=0, y=50, z=0)
        b = Block(color="Blue", x=100, y=50, z=0)
        assert is_touching(a, b, cfg)

    def test_is_touching_stacked(self):
        cfg = GridConfig()
        a = Block(color="Red", x=0, y=50, z=0)
        b = Block(color="Blue", x=0, y=150, z=0)
        assert is_touching(a, b, cfg)

    def test_not_touching(self):
        cfg = GridConfig()
        a = Block(color="Red", x=0, y=50, z=0)
        b = Block(color="Blue", x=200, y=50, z=0)
        assert not is_touching(a, b, cfg)

    def test_relationship_on_top(self):
        cfg = GridConfig()
        a = Block(color="Red", x=0, y=50, z=0)
        b = Block(color="Blue", x=0, y=150, z=0)
        assert relationship(a, b, cfg) == "on top of"

    def test_relationship_right(self):
        cfg = GridConfig()
        a = Block(color="Red", x=0, y=50, z=0)
        b = Block(color="Blue", x=100, y=50, z=0)
        assert relationship(a, b, cfg) == "to the right of"

    def test_is_connected_single(self):
        g = Grid.from_str("Red,0,50,0")
        assert is_connected(g)

    def test_is_connected_touching(self):
        g = Grid.from_str("Red,0,50,0;Blue,100,50,0")
        assert is_connected(g)

    def test_not_connected(self):
        g = Grid.from_str("Red,0,50,0;Blue,300,50,300")
        assert not is_connected(g)


# --- Instruction parser tests ---

class TestInstructionParser:
    def test_parse_full_message(self):
        msg = (
            "[TASK_DESCRIPTION] Grid: 9x9 cells.\n"
            "[SPEAKER] Anna\n"
            "[START_STRUCTURE] Red,0,50,0\n"
            "Stack three blue blocks on top of the red block."
        )
        parsed = parse_green_message(msg)
        assert "9x9" in parsed.task_description
        assert parsed.speaker == "Anna"
        assert parsed.start_structure_str == "Red,0,50,0"
        assert len(parsed.start_grid.blocks) == 1
        assert "Stack" in parsed.instruction_text or "blue" in parsed.instruction_text

    def test_parse_empty_start(self):
        msg = (
            "[TASK_DESCRIPTION] Grid info\n"
            "[SPEAKER] Emma\n"
            "[START_STRUCTURE] \n"
            "Place a red block in each corner."
        )
        parsed = parse_green_message(msg)
        assert parsed.speaker == "Emma"
        assert len(parsed.start_grid.blocks) == 0

    def test_parse_feedback(self):
        msg = "Feedback: correct build | Round score: +10 | Total score: +10"
        parsed = parse_green_message(msg)
        assert parsed.is_feedback
        assert "correct" in parsed.feedback_text

    def test_parse_transition(self):
        msg = "A new task is starting, now you will play the game again."
        parsed = parse_green_message(msg)
        assert parsed.is_feedback


# --- Underspec detector tests (heuristic) ---

class TestUnderspecHeuristic:
    """Tests for detect_underspec_heuristic against real CSV trial patterns."""

    # ── fully_spec: NO flags should fire ──

    def test_fully_spec_two_colors_all_specified(self):
        """Trial 12: both colors named with counts — nothing missing."""
        r = detect_underspec_heuristic(
            "Stack 3 red blocks in the bottom right corner. "
            "Put 2 yellow blocks on top of the red stack you just built."
        )
        assert not r.has_missing_color

    def test_fully_spec_single_color(self):
        r = detect_underspec_heuristic(
            "Place 9 purple blocks along the grid's left edge."
        )
        assert not r.has_missing_color

    # ── color_under: one color mentioned, colorless phrase → MUST ASK ──

    def test_color_under_single_color_must_ask(self):
        """List1 Trial 1a: 'stack 4 blocks in front' — only 'purple' in text.
        Always ask on colorless phrases — asking is +EV at ≥75% build success."""
        r = detect_underspec_heuristic(
            "Stack 5 purple blocks in the middle of the grid, "
            "then stack 4 blocks in front of them."
        )
        assert r.has_missing_color, "Colorless phrase must trigger ASK"
        assert r.inferred_color == "Purple"  # fallback still set
        assert r.suggested_question

    def test_color_under_single_color_must_ask_green(self):
        """List2 Trial 2a: 'stack 3 blocks immediately to the left' — only 'green' in text."""
        r = detect_underspec_heuristic(
            "Place a green block on the existing green block. "
            "Then stack 3 blocks immediately to the left of these."
        )
        assert r.has_missing_color
        assert r.inferred_color == "Green"
        assert r.suggested_question

    def test_color_under_single_color_must_ask_blue(self):
        """List1 Trial 18a: 'stack 2 blocks to the left' — only 'blue' in text."""
        r = detect_underspec_heuristic(
            "Stack 3 blue blocks in front of the existing blue blocks. "
            "Then stack 2 blocks to the left of the tower you just built."
        )
        assert r.has_missing_color
        assert r.inferred_color == "Blue"
        assert r.suggested_question

    # ── color_under with multiple colors → MUST ASK ──

    def test_color_under_two_colors_must_ask(self):
        """List1 Trial 2a: 'red' and 'yellow' named, but 'Build a tower of 4 blocks'
        has no color. Missing color = Blue (unpredictable). Must ask."""
        r = detect_underspec_heuristic(
            "Stack 3 red blocks to the left of the yellow block. "
            "Build a tower of 4 blocks in front of these."
        )
        assert r.has_missing_color, (
            "Multiple colors + colorless phrase → must ask, not infer"
        )
        assert r.suggested_question

    # ── color_under: no colors at all → MUST ASK ──

    def test_no_colors_at_all(self):
        r = detect_underspec_heuristic(
            "Stack blocks on the highlighted square."
        )
        assert r.has_missing_color
        assert r.suggested_question

    # ── number_under: always infer, never ask ──

    def test_number_under_infer_count(self):
        """List1 Trial 6a: 'Build a red stack to the right of the blue one.'
        Missing count should be inferred, not asked."""
        r = detect_underspec_heuristic(
            "Build a row of 3 green blocks, starting from the middle square. "
            "Build a blue stack of 3 blocks immediately to the right. "
            "Build a red stack to the right of the blue one."
        )
        assert not r.has_missing_color  # all colors specified
        assert r.has_missing_number
        assert r.inferred_count > 0
        # Should NOT suggest a question for count
        assert "color" not in r.suggested_question.lower() if r.suggested_question else True

    def test_number_only_generates_count_question(self):
        """When only count is missing (all colors specified), generate count question."""
        r = detect_underspec_heuristic(
            "Build a row of 3 green blocks, starting from the middle square. "
            "Build a red stack to the right of the green one."
        )
        assert not r.has_missing_color
        assert r.has_missing_number
        assert r.suggested_question == "", (
            "Number-only underspec must NOT generate a color question"
        )
        assert r.suggested_count_question, (
            "Number-only underspec must generate a count question"
        )

    # ── combined color+number underspec → single question, color only ──

    def test_combined_color_number_asks_color_only(self):
        """When both color AND number are missing, only ask about color
        via suggested_question. Count question goes in suggested_count_question.
        The pipeline prioritizes color over count (one question per round)."""
        r = detect_underspec_heuristic(
            "Stack blocks on the highlighted square."
        )
        assert r.has_missing_color
        assert r.has_missing_number
        assert r.suggested_question  # must have a color question
        assert "color" in r.suggested_question.lower()
        assert "how many" not in r.suggested_question.lower(), (
            "Color question must NOT ask about count"
        )
        # Count question available separately for pipeline decision
        assert r.suggested_count_question

    def test_single_question_only(self):
        """Regardless of how many gaps, at most one suggested_question."""
        r = detect_underspec_heuristic(
            "Stack purple blocks on the highlighted square. "
            "Then place blocks to the right."
        )
        # Should have one question about color, not multiple
        assert r.suggested_question.count("?") == 1, (
            f"Expected exactly one question, got: {r.suggested_question!r}"
        )

    # ── detect_underspec_from_plan: same single-question rules ──

    def test_from_plan_color_only_question(self):
        """detect_underspec_from_plan should only ask about color, never count."""
        plan = [
            {"action": "stack", "color": "Red", "count": 3},
            {"action": "stack", "color": "Blue", "count": 2},
            {"action": "stack", "color": "Uncolored", "count": "Uncounted"},
        ]
        r = detect_underspec_from_plan(plan)
        assert r.has_missing_color
        assert r.has_missing_number
        assert r.suggested_question
        assert "how many" not in r.suggested_question.lower(), (
            "Must NOT ask about count — only color"
        )

    def test_from_plan_number_only_no_question(self):
        """detect_underspec_from_plan: count-only gap should not generate a question."""
        plan = [
            {"action": "stack", "color": "Red", "count": 3},
            {"action": "stack", "color": "Blue", "count": "Uncounted"},
        ]
        r = detect_underspec_from_plan(plan)
        assert not r.has_missing_color
        assert r.has_missing_number
        assert r.suggested_question == "", (
            "Number-only underspec from plan must not generate a question"
        )


# --- CSV-driven heuristic validation (all data/ samples) ---

def _load_csv_instructions():
    """Load all unique (instruction, trialType) from data/*.csv files."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    seen = set()
    results = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv"):
            continue
        with open(os.path.join(data_dir, fname)) as f:
            for row in csv.DictReader(f):
                tt = row["trialType"]
                for col in ("sentenceW", "sentenceD"):
                    instr = row[col]
                    key = (instr, tt)
                    if key not in seen:
                        seen.add(key)
                        results.append((instr, tt, fname.split("_")[0]))
    return results


_CSV_DATA = _load_csv_instructions()


def _csv_params(trial_type):
    """Build pytest.param list for a given trial type."""
    return [
        pytest.param(instr, id=f"{src}|{instr[:50]}")
        for instr, tt, src in _CSV_DATA
        if tt == trial_type
    ]


class TestHeuristicAllCSVData:
    """Validate detect_underspec_heuristic against EVERY instruction from data/ CSVs.

    Covers both sentenceW (word-numbers) and sentenceD (digit-numbers) forms.
    Ensures the heuristic produces zero false positives on the real stimuli.
    """

    @pytest.mark.parametrize("instruction", _csv_params("fully_spec"))
    def test_fully_spec_no_false_positive(self, instruction):
        """fully_spec instructions: has_missing_color must be False."""
        r = detect_underspec_heuristic(instruction)
        assert not r.has_missing_color, (
            f"False positive on fully_spec:\n  {instruction!r}\n  {r.details}"
        )

    @pytest.mark.parametrize("instruction", _csv_params("color_under"))
    def test_color_under_always_asks(self, instruction):
        """color_under: ALWAYS ask — any colorless phrase must trigger ASK."""
        r = detect_underspec_heuristic(instruction)
        assert r.has_missing_color, (
            f"color_under should ALWAYS ask:\n  {instruction!r}\n  {r.details}"
        )
        assert r.suggested_question

    @pytest.mark.parametrize("instruction", _csv_params("number_under"))
    def test_number_under_no_color_false_positive(self, instruction):
        """number_under: all colors specified -> has_missing_color must be False."""
        r = detect_underspec_heuristic(instruction)
        assert not r.has_missing_color, (
            f"False positive on number_under:\n  {instruction!r}\n  {r.details}"
        )

    @pytest.mark.parametrize("instruction", _csv_params("number_under"))
    def test_number_under_detects_missing_count(self, instruction):
        """number_under: should detect a missing count and infer a positive value."""
        r = detect_underspec_heuristic(instruction)
        assert r.has_missing_number, (
            f"Should detect missing count:\n  {instruction!r}"
        )
        assert r.inferred_count > 0

    @pytest.mark.parametrize("instruction", _csv_params("number_under"))
    def test_number_under_generates_count_question(self, instruction):
        """number_under: must produce a suggested_count_question."""
        r = detect_underspec_heuristic(instruction)
        assert r.suggested_count_question, (
            f"Should generate count question:\n  {instruction!r}"
        )

    @pytest.mark.parametrize("instruction", _csv_params("fully_spec"))
    def test_fully_spec_no_count_false_positive(self, instruction):
        """fully_spec: has_missing_number must be False (no spurious count-asks)."""
        r = detect_underspec_heuristic(instruction)
        assert not r.has_missing_number, (
            f"Count false positive on fully_spec:\n  {instruction!r}\n  {r.details}"
        )

    @pytest.mark.parametrize("instruction", _csv_params("color_under"))
    def test_color_under_no_count_false_positive(self, instruction):
        """color_under: has_missing_number must be False (no spurious count-asks)."""
        r = detect_underspec_heuristic(instruction)
        assert not r.has_missing_number, (
            f"Count false positive on color_under:\n  {instruction!r}\n  {r.details}"
        )


class TestPatchAllColorUnder:
    """Verify patch_instruction_with_color on every color_under instruction."""

    @pytest.mark.parametrize("instruction", _csv_params("color_under"))
    def test_patch_preserves_existing_colors(self, instruction):
        """Original color words must survive patching."""
        from skills.underspec_detector import _extract_colors_from_text
        original_colors = _extract_colors_from_text(instruction)
        patched = patch_instruction_with_color(instruction, "Purple")
        patched_colors = _extract_colors_from_text(patched)
        for c in original_colors:
            assert c in patched_colors, (
                f"Lost color {c!r}:\n  orig: {instruction!r}\n  patched: {patched!r}"
            )

    @pytest.mark.parametrize("instruction", _csv_params("color_under"))
    def test_patch_never_shrinks(self, instruction):
        """Patching only inserts color words -- text never gets shorter."""
        patched = patch_instruction_with_color(instruction, "Green")
        assert len(patched) >= len(instruction)


# --- Spatial executor tests ---

class TestSpatialExecutor:
    def test_stack_at_origin(self):
        grid = Grid()
        executor = SpatialExecutor(grid)
        step = BuildStep(
            action="stack",
            color="Red",
            count=3,
            position={"x": 0, "z": 0},
        )
        executor.execute_step(step)
        assert len(grid.blocks) == 3
        assert grid.blocks[0].y == 50
        assert grid.blocks[1].y == 150
        assert grid.blocks[2].y == 250

    def test_place_single(self):
        grid = Grid()
        executor = SpatialExecutor(grid)
        step = BuildStep(
            action="place",
            color="Blue",
            count=1,
            position={"x": 100, "z": 200},
        )
        executor.execute_step(step)
        assert len(grid.blocks) == 1
        assert grid.blocks[0].x == 100
        assert grid.blocks[0].z == 200

    def test_place_relative_right(self):
        grid = Grid.from_str("Red,0,50,0")
        executor = SpatialExecutor(grid)
        step = BuildStep(
            action="place_relative",
            color="Blue",
            count=1,
            position={
                "relative_to": "existing_Red_stack_at_0_0",
                "direction": "right",
                "distance": 1,
            },
        )
        executor.execute_step(step)
        assert len(grid.blocks) == 2
        assert grid.blocks[1].x == 100
        assert grid.blocks[1].z == 0

    def test_place_at_corners(self):
        grid = Grid()
        executor = SpatialExecutor(grid)
        step = BuildStep(
            action="place_at_corners",
            color="Red",
            count=4,
            position={},
        )
        executor.execute_step(step)
        assert len(grid.blocks) == 4
        positions = {(b.x, b.z) for b in grid.blocks}
        assert (-400, -400) in positions
        assert (400, -400) in positions
        assert (-400, 400) in positions
        assert (400, 400) in positions

    def test_extend_row(self):
        grid = Grid.from_str("Red,0,50,0")
        executor = SpatialExecutor(grid)
        step = BuildStep(
            action="extend_row",
            color="Blue",
            count=3,
            position={"x": 0, "z": 0, "direction": "right"},
        )
        executor.execute_step(step)
        assert len(grid.blocks) == 4  # 1 original + 3 new
        xs = sorted(b.x for b in grid.blocks if b.color == "Blue")
        # First block is placed AT start position (0,0), then 100, 200
        assert xs == [0, 100, 200]

    def test_extend_row_relative_to_rightmost(self):
        """Extend row from rightmost block should place NEXT TO it, not on top."""
        grid = Grid.from_str("Red,-100,50,0;Red,0,50,0;Red,100,50,0")
        executor = SpatialExecutor(grid)
        step = BuildStep(
            action="extend_row",
            color="Red",
            count=2,
            position={"relative_to": "rightmost", "direction": "right"},
        )
        executor.execute_step(step)
        assert len(grid.blocks) == 5  # 3 original + 2 new
        xs = sorted(b.x for b in grid.blocks)
        assert xs == [-100, 0, 100, 200, 300]
        # All blocks should be at ground level (y=50)
        ys = [b.y for b in grid.blocks]
        assert all(y == 50 for y in ys), f"Expected all y=50, got {ys}"

    def test_extend_row_then_stack_on_ends(self):
        """Full trial 4b: extend red row + stack on each end."""
        grid = Grid.from_str("Red,-100,50,0;Red,0,50,0;Red,100,50,0")
        executor = SpatialExecutor(grid)
        steps = [
            BuildStep(
                action="extend_row",
                color="Red",
                count=2,
                position={"relative_to": "rightmost", "direction": "right"},
            ),
            BuildStep(
                action="stack",
                color="Red",
                count=1,
                position={"relative_to": "rightmost", "direction": "on_top"},
            ),
            BuildStep(
                action="stack",
                color="Red",
                count=1,
                position={"relative_to": "leftmost", "direction": "on_top"},
            ),
        ]
        executor.execute_plan(steps)
        assert len(grid.blocks) == 7
        expected_positions = {
            (-100, 50, 0), (0, 50, 0), (100, 50, 0),
            (200, 50, 0), (300, 50, 0),
            (300, 150, 0), (-100, 150, 0),
        }
        actual_positions = {(b.x, b.y, b.z) for b in grid.blocks}
        assert actual_positions == expected_positions, (
            f"Expected {expected_positions}, got {actual_positions}"
        )

    def test_uncolored_defaults(self):
        """Uncolored is resolved to a default rather than raising."""
        grid = Grid()
        executor = SpatialExecutor(grid)
        step = BuildStep(
            action="stack",
            color="Uncolored",
            count=3,
            position={"x": 0, "z": 0},
        )
        # Should NOT raise — resolves to default color
        executor.execute_step(step)
        assert len(grid.blocks) == 3

    def test_execute_plan_trial_9(self):
        """Trial 9: Place a red block in each corner, then green on top of each."""
        grid = Grid()
        executor = SpatialExecutor(grid)
        steps = [
            BuildStep(action="place_at_corners", color="Red", count=4, position={}),
            BuildStep(action="stack", color="Green", count=1, position={"x": -400, "z": -400}),
            BuildStep(action="stack", color="Green", count=1, position={"x": 400, "z": -400}),
            BuildStep(action="stack", color="Green", count=1, position={"x": 400, "z": 400}),
            BuildStep(action="stack", color="Green", count=1, position={"x": -400, "z": 400}),
        ]
        executor.execute_plan(steps)
        assert len(grid.blocks) == 8
        # All corners should have 2 blocks (red + green)
        for corner_xz in [(-400, -400), (400, -400), (400, 400), (-400, 400)]:
            stack = grid.blocks_at_xz(*corner_xz)
            assert len(stack) == 2
            assert stack[0].color == "Red"
            assert stack[1].color == "Green"


# --- Response formatter tests ---

class TestResponseFormatter:
    def test_format_empty(self):
        g = Grid()
        resp = format_build_response(g)
        assert resp == "[BUILD]"

    def test_format_blocks(self):
        g = Grid.from_str("Red,0,50,0;Blue,100,50,0")
        resp = format_build_response(g)
        assert resp.startswith("[BUILD]")
        assert "Red,0,50,0" in resp
        assert "Blue,100,50,0" in resp

    def test_validate_valid(self):
        is_valid, errors = validate_build_response("[BUILD];Red,0,50,0;Blue,100,50,0")
        assert is_valid
        assert len(errors) == 0

    def test_validate_invalid_prefix(self):
        is_valid, errors = validate_build_response("Coordinates:Red,0,50,0")
        assert not is_valid

    def test_validate_invalid_coords(self):
        is_valid, errors = validate_build_response("[BUILD];Red,50,50,50")
        assert not is_valid  # x=50 is not valid

    def test_validate_duplicate_pos(self):
        is_valid, errors = validate_build_response("[BUILD];Red,0,50,0;Blue,0,50,0")
        assert not is_valid


# --- Multi-color answer disambiguation tests ---

class TestMultiColorDisambiguation:
    """Test the logic for picking the correct color from multi-color answers.

    When the green agent answers "Blue and Green", we must filter out
    colors already in the instruction and use the remaining (new) color.
    """
    COLOR_NAMES = {"red", "blue", "green", "yellow", "purple", "orange",
                   "white", "black", "brown", "pink", "grey", "gray", "cyan"}

    @staticmethod
    def _disambiguate(answered_colors, instruction):
        """Replicate the disambiguation logic from server.py."""
        color_names = {"red", "blue", "green", "yellow", "purple", "orange",
                       "white", "black", "brown", "pink", "grey", "gray", "cyan"}
        instruction_lower = instruction.lower()
        instruction_colors = {c for c in color_names if c in instruction_lower}

        if len(answered_colors) > 1:
            new_colors = [
                c for c in answered_colors
                if c.lower() not in instruction_colors
            ]
            return new_colors[0] if new_colors else answered_colors[-1]
        return answered_colors[0]

    def test_two_colors_first_matches_instruction(self):
        # "stack five purple blocks... stack four blocks" → Answer: Purple, Yellow
        # Purple is already in instruction → use Yellow
        result = self._disambiguate(
            ["Purple", "Yellow"],
            "Stack five purple blocks in the middle of the grid, then stack four blocks in front of them."
        )
        assert result == "Yellow"

    def test_two_colors_first_matches_reversed(self):
        # Same but answer order reversed
        result = self._disambiguate(
            ["Yellow", "Purple"],
            "Stack five purple blocks in the middle of the grid, then stack four blocks in front of them."
        )
        assert result == "Yellow"

    def test_two_colors_second_matches_instruction(self):
        # "three yellow blocks... Stack two blocks" → Answer: Yellow, Purple
        # Yellow is already in instruction → use Purple
        result = self._disambiguate(
            ["Yellow", "Purple"],
            "place a horizontal row of three yellow blocks going left. Stack two blocks on top."
        )
        assert result == "Purple"

    def test_two_colors_blue_and_green_with_blue_instruction(self):
        # "Stack three blue blocks... stack two blocks to the left"
        # Answer: Blue, Green → Blue in instruction → use Green
        result = self._disambiguate(
            ["Blue", "Green"],
            "Stack three blue blocks in front of the existing blue blocks. Then stack two blocks to the left."
        )
        assert result == "Green"

    def test_single_color_answer(self):
        # Single color → just use it
        result = self._disambiguate(
            ["Red"],
            "Stack blocks on the highlighted square."
        )
        assert result == "Red"

    def test_two_colors_neither_in_instruction(self):
        # No colors in instruction → use the first answered color
        result = self._disambiguate(
            ["Blue", "Red"],
            "Stack blocks in the middle, then build a stack to the left."
        )
        assert result == "Blue"

    def test_two_colors_both_in_instruction(self):
        # Both colors already in instruction → fall back to last
        result = self._disambiguate(
            ["Blue", "Red"],
            "Stack blue blocks on the left. Stack red blocks on the right. Add blocks on top."
        )
        assert result == "Red"


class TestAnswerCountExtraction:
    """Test _extract_answer_count from OpenAIPurpleAgent."""

    @staticmethod
    def _extract(text):
        from purple_openai.server import OpenAIPurpleAgent
        return OpenAIPurpleAgent._extract_answer_count(text)

    def test_digit_with_blocks(self):
        assert self._extract("Answer: 4 blocks (-5 points for asking)") == 4

    def test_digit_alone(self):
        assert self._extract("Answer: 2 (-5 points for asking)") == 2

    def test_word_number(self):
        assert self._extract("Answer: three (-5 points for asking)") == 3

    def test_word_five(self):
        assert self._extract("Answer: five blocks (-5 points for asking)") == 5

    def test_blocks_high(self):
        assert self._extract("Answer: 3 blocks high (-5 points for asking)") == 3

    def test_no_answer_pattern(self):
        assert self._extract("Hello there") is None

    def test_color_answer_returns_none(self):
        # Color answers should not return a number
        assert self._extract("Answer: Blue (-5 points for asking)") is None

    def test_color_and_number_returns_number(self):
        # "4 blocks" has a number even if it looks unusual
        assert self._extract("Answer: 4 blue blocks (-5 points)") == 4


class TestPatchInstructionWithCount:
    """Test patch_instruction_with_count."""

    @staticmethod
    def _patch(instruction, count, target_color=""):
        from skills.underspec_detector import patch_instruction_with_count
        return patch_instruction_with_count(instruction, count, target_color)

    def test_simple_color_stack(self):
        result = self._patch("Build a blue stack in front of the yellow one.", 3)
        assert "3" in result
        assert "blue" in result.lower()
        assert "blocks" in result.lower()

    def test_finish_with_stack(self):
        result = self._patch("Finish with a yellow stack on the left.", 4, "yellow")
        assert "4" in result
        assert "yellow" in result.lower()

    def test_plural_blocks(self):
        result = self._patch("Stack blue blocks to the right.", 4)
        assert "4 blue blocks" in result.lower()

    def test_no_change_when_already_counted(self):
        original = "Build a stack of three blue blocks."
        result = self._patch(original, 5)
        assert result == original

    def test_target_color_filter(self):
        # Only patch the yellow phrase, not the red
        instr = "Build a red stack to the right. Build a yellow stack to the left."
        result = self._patch(instr, 4, "yellow")
        assert "4" in result
        # Red stack should stay unpatched
        assert "red stack" in result.lower()

    def test_multiple_uncounted(self):
        instr = "Build a red stack to the right. Build a yellow stack to the left."
        result = self._patch(instr, 3)
        # Both should get patched
        assert result.lower().count("3") >= 2


class TestColorSpecificCountQuestion:
    """Test that count questions include the color of the uncounted phrase."""

    @staticmethod
    def _detect(instruction):
        from skills.underspec_detector import detect_underspec_heuristic
        return detect_underspec_heuristic(instruction)

    def test_blue_stack_question(self):
        r = self._detect("Build three yellow blocks. Build a blue stack in front.")
        assert r.has_missing_number
        assert r.uncounted_color == "Blue"
        assert "blue" in r.suggested_count_question.lower()

    def test_no_color_fallback(self):
        r = self._detect("Build three yellow blocks. Build a stack in front.")
        assert r.has_missing_number
        # No color in the uncounted phrase -> generic question
        if not r.uncounted_color:
            assert "unspecified" in r.suggested_count_question.lower()

    def test_multiple_uncounted_uses_first_color(self):
        r = self._detect("Build a red stack. Build a yellow stack.")
        assert r.has_missing_number
        assert len(r.uncounted_phrases) >= 2
        # First color should be used
        assert r.uncounted_color in ("Red", "Yellow")

    def test_yellow_stack_of_three_targets_green(self):
        """'a yellow stack of three blocks' has a specified count.
        The unspecified stack is the green one, so the question should
        reference green, not yellow."""
        r = self._detect(
            "Build a yellow stack of three blocks to the left of the "
            "existing block. Build a green stack to the left of the "
            "yellow stack."
        )
        assert r.has_missing_number
        assert r.uncounted_color == "Green", (
            f"Question should target Green (unspecified), not "
            f"'{r.uncounted_color}'"
        )
        assert "green" in r.suggested_count_question.lower()

    def test_blue_stack_of_three_targets_red(self):
        """'a blue stack of three blocks' has a specified count.
        The unspecified stack is the red one."""
        r = self._detect(
            "Build a row of three green blocks, starting from the "
            "middle square and going to the right. Build a blue stack "
            "of three blocks immediately to the right of the green "
            "row. Build a red stack to the right of the blue one."
        )
        assert r.has_missing_number
        assert r.uncounted_color == "Red", (
            f"Question should target Red (unspecified), not "
            f"'{r.uncounted_color}'"
        )
        assert "red" in r.suggested_count_question.lower()

    def test_count_after_noun_not_treated_as_missing(self):
        """Phrases like 'a yellow stack of three blocks' should not
        appear in uncounted_phrases."""
        r = self._detect(
            "Build a yellow stack of three blocks. "
            "Build a green stack nearby."
        )
        # Yellow should NOT be in uncounted_phrases
        uncounted_colors = [c for c, _ in r.uncounted_phrases]
        assert "Yellow" not in uncounted_colors, (
            "Yellow has a specified count ('of three blocks') and "
            "should not be in uncounted_phrases"
        )


class TestCompoundQuestion:
    """Test compound question generation when both color and count are missing."""

    @staticmethod
    def _detect(instruction):
        from skills.underspec_detector import detect_underspec_heuristic
        return detect_underspec_heuristic(instruction)

    def test_both_missing_generates_compound(self):
        """When both color and count are missing, a compound question is generated."""
        r = self._detect(
            "Stack three red blocks to the left. "
            "Build a tower in front of these."
        )
        assert r.has_missing_color
        assert r.has_missing_number
        assert r.suggested_compound_question
        assert "color" in r.suggested_compound_question.lower()
        assert "how many" in r.suggested_compound_question.lower()

    def test_color_only_no_compound(self):
        """When only color is missing, no compound question."""
        r = self._detect(
            "Build a stack of three blocks on the highlighted square."
        )
        if r.has_missing_color and not r.has_missing_number:
            assert r.suggested_compound_question == ""

    def test_count_only_no_compound(self):
        """When only count is missing, no compound question."""
        r = self._detect(
            "Build three yellow blocks. Build a blue stack in front."
        )
        if r.has_missing_number and not r.has_missing_color:
            assert r.suggested_compound_question == ""

    def test_neither_missing_no_compound(self):
        """When nothing is missing, no compound question."""
        r = self._detect(
            "Build a stack of three blue blocks on the highlighted square."
        )
        assert r.suggested_compound_question == ""

    def test_compound_question_format(self):
        """Compound question asks about both color and count."""
        r = self._detect(
            "Build a tower of three blue blocks. Build a tower in front."
        )
        if r.has_missing_color and r.has_missing_number:
            q = r.suggested_compound_question
            assert "color" in q.lower() or "what color" in q.lower()
            assert "how many" in q.lower() or "blocks" in q.lower()


# --- Auto-fix each-end caps tests ---

from skills.plan_verifier import auto_fix_each_end_caps


class TestAutoFixEachEndCaps:
    """Tests for deterministic each-end cap position correction."""

    def _make_grid(self, blocks_str: str) -> Grid:
        return Grid.from_str(blocks_str)

    def _make_step(self, action: str, color: str, count: int, pos: dict) -> BuildStep:
        return BuildStep(action=action, color=color, count=count, position=pos)

    def test_fixes_right_end_cap_after_extend_right(self):
        """After extending right, cap at old right end should move to new right end."""
        instruction = "Extend it horizontally by adding two red blocks to its right. Place one block on top of each end of this extended row."
        grid = self._make_grid("Red,-100,50,0;Red,0,50,0;Red,100,50,0")
        steps = [
            self._make_step("extend_row", "Red", 2, {"x": 200, "z": 0, "direction": "right"}),
            self._make_step("stack", "Red", 1, {"x": -100, "z": 0}),   # left end (correct)
            self._make_step("stack", "Red", 1, {"x": 100, "z": 0}),     # old right end (WRONG)
        ]
        fixed = auto_fix_each_end_caps(instruction, steps, grid)
        assert int(fixed[1].position["x"]) == -100, "Left end should stay at -100"
        assert int(fixed[2].position["x"]) == 300, "Right end should be fixed to 300"

    def test_no_change_when_already_correct(self):
        """If the cap is already at the correct new end, nothing changes."""
        instruction = "Place one block on top of each end of this extended row."
        grid = self._make_grid("Red,-100,50,0;Red,0,50,0;Red,100,50,0")
        steps = [
            self._make_step("extend_row", "Red", 2, {"x": 200, "z": 0, "direction": "right"}),
            self._make_step("stack", "Red", 1, {"x": -100, "z": 0}),
            self._make_step("stack", "Red", 1, {"x": 300, "z": 0}),     # already correct
        ]
        fixed = auto_fix_each_end_caps(instruction, steps, grid)
        assert int(fixed[2].position["x"]) == 300

    def test_no_trigger_without_each_end(self):
        """If instruction doesn't mention 'each end', no fix is applied."""
        instruction = "Extend it horizontally by adding two red blocks to its right."
        grid = self._make_grid("Red,-100,50,0;Red,0,50,0;Red,100,50,0")
        steps = [
            self._make_step("extend_row", "Red", 2, {"x": 200, "z": 0, "direction": "right"}),
            self._make_step("stack", "Red", 1, {"x": 100, "z": 0}),
        ]
        fixed = auto_fix_each_end_caps(instruction, steps, grid)
        assert int(fixed[1].position["x"]) == 100, "Should not change without 'each end'"

    def test_fixes_left_end_cap_after_extend_left(self):
        """After extending left, cap at old left end should move to new left end."""
        instruction = "Extend it by adding two blocks to its left. Place one block on top of each end."
        grid = self._make_grid("Red,-100,50,0;Red,0,50,0;Red,100,50,0")
        steps = [
            self._make_step("extend_row", "Red", 2, {"x": -200, "z": 0, "direction": "left"}),
            self._make_step("stack", "Red", 1, {"x": -100, "z": 0}),    # old left end (WRONG)
            self._make_step("stack", "Red", 1, {"x": 100, "z": 0}),     # right end (correct)
        ]
        fixed = auto_fix_each_end_caps(instruction, steps, grid)
        assert int(fixed[1].position["x"]) == -300, "Left end should be fixed to -300"
        assert int(fixed[2].position["x"]) == 100, "Right end should stay at 100"

    def test_both_ends_trigger(self):
        """'both ends' should also trigger the fix."""
        instruction = "Place a block on both ends of the extended row."
        grid = self._make_grid("Red,-100,50,0;Red,0,50,0;Red,100,50,0")
        steps = [
            self._make_step("extend_row", "Red", 2, {"x": 200, "z": 0, "direction": "right"}),
            self._make_step("stack", "Red", 1, {"x": -100, "z": 0}),
            self._make_step("stack", "Red", 1, {"x": 100, "z": 0}),
        ]
        fixed = auto_fix_each_end_caps(instruction, steps, grid)
        assert int(fixed[2].position["x"]) == 300

    def test_no_extend_step_returns_unchanged(self):
        """If there is no extend_row step and no extension phrase in instruction, nothing changes."""
        instruction = "Place one block on top of each end."
        grid = self._make_grid("Red,-100,50,0;Red,0,50,0;Red,100,50,0")
        steps = [
            self._make_step("stack", "Red", 1, {"x": -100, "z": 0}),
            self._make_step("stack", "Red", 1, {"x": 100, "z": 0}),
        ]
        fixed = auto_fix_each_end_caps(instruction, steps, grid)
        assert int(fixed[0].position["x"]) == -100
        assert int(fixed[1].position["x"]) == 100

    def test_fallback_no_extend_row_but_instruction_says_extend_right(self):
        """Fallback: no extend_row step, but instruction says 'adding two blocks to its right'."""
        instruction = "Extend it horizontally by adding two red blocks to its right. Place one block on top of each end of this extended row."
        grid = self._make_grid("Red,-100,50,0;Red,0,50,0;Red,100,50,0")
        # LLM used individual place steps instead of extend_row
        steps = [
            self._make_step("place", "Red", 1, {"x": 200, "z": 0}),
            self._make_step("place", "Red", 1, {"x": 300, "z": 0}),
            self._make_step("stack", "Red", 1, {"x": -100, "z": 0}),   # left cap (correct)
            self._make_step("stack", "Red", 1, {"x": 100, "z": 0}),     # old right cap (WRONG)
        ]
        fixed = auto_fix_each_end_caps(instruction, steps, grid)
        assert int(fixed[2].position["x"]) == -100, "Left end should stay at -100"
        assert int(fixed[3].position["x"]) == 300, "Right end should be fixed to 300 via fallback"

    def test_fallback_no_extend_row_extend_left(self):
        """Fallback: instruction says 'adding three blocks going left'."""
        instruction = "Add three green blocks going left. Put a block on each end."
        grid = self._make_grid("Green,100,50,0;Green,200,50,0")
        steps = [
            self._make_step("place", "Green", 1, {"x": 0, "z": 0}),
            self._make_step("place", "Green", 1, {"x": -100, "z": 0}),
            self._make_step("place", "Green", 1, {"x": -200, "z": 0}),
            self._make_step("stack", "Green", 1, {"x": 200, "z": 0}),   # right end (correct)
            self._make_step("stack", "Green", 1, {"x": 100, "z": 0}),   # old left end (WRONG)
        ]
        fixed = auto_fix_each_end_caps(instruction, steps, grid)
        assert int(fixed[3].position["x"]) == 200, "Right end should stay at 200"
        assert int(fixed[4].position["x"]) == -200, "Left end should be fixed to -200 via fallback"

    def test_z_axis_extend_front(self):
        """Test z-axis row extension (front direction)."""
        instruction = "Extend it by adding two blocks to the front. Place one on each end."
        grid = self._make_grid("Red,0,50,-100;Red,0,50,0;Red,0,50,100")
        steps = [
            self._make_step("extend_row", "Red", 2, {"x": 0, "z": 200, "direction": "front"}),
            self._make_step("stack", "Red", 1, {"x": 0, "z": -100}),    # unchanged end (correct)
            self._make_step("stack", "Red", 1, {"x": 0, "z": 100}),     # old front end (WRONG)
        ]
        fixed = auto_fix_each_end_caps(instruction, steps, grid)
        assert int(fixed[1].position["z"]) == -100
        assert int(fixed[2].position["z"]) == 300, "Front end should be fixed to 300"


# --- Auto-fix T-shape extend tests ---

from skills.plan_verifier import auto_fix_t_shape_extend


class TestAutoFixTShapeExtend:
    """Tests for deterministic T-shape extend direction correction."""

    def _make_grid(self, blocks_str: str) -> Grid:
        return Grid.from_str(blocks_str)

    def _make_step(self, action: str, color: str, count: int, pos: dict) -> BuildStep:
        return BuildStep(action=action, color=color, count=count, position=pos)

    def test_fixes_on_top_to_front_for_stem_along_plus_z(self):
        """Actual stimulus: stem runs +z, model says 'on_top', should be 'front'."""
        instruction = "Keeping the T shape, extend the existing green structure by adding two green blocks to the longer base."
        # Crossbar at z=-100: (-100,-100), (0,-100), (100,-100)
        # Stem along z at x=0: (0,-100), (0,0), (0,100), (0,200)
        grid = self._make_grid(
            "Green,-100,50,-100;Green,0,50,-100;Green,100,50,-100;"
            "Green,0,50,0;Green,0,50,100;Green,0,50,200"
        )
        steps = [
            self._make_step("extend_row", "Green", 2, {"x": 0, "z": 200, "direction": "on_top"}),
            self._make_step("place", "Purple", 1, {"x": -200, "z": -100}),
            self._make_step("place", "Purple", 1, {"x": 200, "z": -100}),
        ]
        fixed = auto_fix_t_shape_extend(instruction, steps, grid)
        assert fixed[0].position["direction"] == "front"
        assert int(fixed[0].position["x"]) == 0
        assert int(fixed[0].position["z"]) == 300  # past base z=200

    def test_fixes_on_top_to_behind_for_stem_along_minus_z(self):
        """Stem runs -z, should map to 'behind'."""
        instruction = "Keeping the T shape, extend by two blocks to the longer base."
        # Crossbar at z=0: (100,0), (200,0), (300,0)
        # Stem along z at x=200: (200,0), (200,-100), (200,-200), (200,-300) = 4 blocks
        grid = self._make_grid(
            "Red,100,50,0;Red,200,50,0;Red,300,50,0;"
            "Red,200,50,-100;Red,200,50,-200;Red,200,50,-300"
        )
        steps = [
            self._make_step("extend_row", "Red", 2, {"x": 200, "z": -300, "direction": "on_top"}),
        ]
        fixed = auto_fix_t_shape_extend(instruction, steps, grid)
        assert fixed[0].position["direction"] == "behind"
        assert int(fixed[0].position["z"]) == -400  # past base z=-300

    def test_no_trigger_without_t_shape(self):
        """If instruction doesn't mention T-shape, no fix."""
        instruction = "Extend the row by two blocks."
        grid = self._make_grid("Red,-100,50,0;Red,0,50,0;Red,100,50,0")
        steps = [
            self._make_step("extend_row", "Red", 2, {"x": 200, "z": 0, "direction": "on_top"}),
        ]
        fixed = auto_fix_t_shape_extend(instruction, steps, grid)
        assert fixed[0].position["direction"] == "on_top"  # not changed

    def test_no_change_when_direction_already_correct(self):
        """If direction is already correct, don't change anything."""
        instruction = "Keeping the T shape, extend by two blocks."
        grid = self._make_grid(
            "Green,-100,50,-100;Green,0,50,-100;Green,100,50,-100;"
            "Green,0,50,0;Green,0,50,100;Green,0,50,200"
        )
        steps = [
            self._make_step("extend_row", "Green", 2, {"x": 0, "z": 300, "direction": "front"}),
        ]
        fixed = auto_fix_t_shape_extend(instruction, steps, grid)
        assert fixed[0].position["direction"] == "front"
        assert int(fixed[0].position["z"]) == 300  # unchanged

    def test_fixes_wrong_direction_right_to_left(self):
        """Stem along x decreasing, model says 'right', should be 'left'."""
        instruction = "Keeping the T shape, extend the longer base."
        # Crossbar at z=0 (vertical): (0,-100), (0,0), (0,100)
        # Stem along x at z=0: (0,0), (-100,0), (-200,0), (-300,0) = 4 blocks
        grid = self._make_grid(
            "Red,0,50,-100;Red,0,50,0;Red,0,50,100;"
            "Red,-100,50,0;Red,-200,50,0;Red,-300,50,0"
        )
        steps = [
            self._make_step("extend_row", "Red", 2, {"x": -300, "z": 0, "direction": "right"}),
        ]
        fixed = auto_fix_t_shape_extend(instruction, steps, grid)
        assert fixed[0].position["direction"] == "left"
        assert int(fixed[0].position["x"]) == -400  # past base x=-300

    def test_no_extend_step_returns_unchanged(self):
        """If no extend_row step, nothing changes."""
        instruction = "Keeping the T shape, add blocks."
        grid = self._make_grid(
            "Green,-100,50,-100;Green,0,50,-100;Green,100,50,-100;"
            "Green,0,50,0;Green,0,50,100;Green,0,50,200"
        )
        steps = [
            self._make_step("place", "Green", 1, {"x": 0, "z": 300}),
        ]
        fixed = auto_fix_t_shape_extend(instruction, steps, grid)
        assert fixed[0].action == "place"
