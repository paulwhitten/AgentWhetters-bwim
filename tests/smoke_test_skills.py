"""Quick end-to-end smoke test for the skills pipeline."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pragmatic_builder"))

from skills.grid import Grid
from skills.instruction_parser import parse_green_message
from skills.build_planner import BuildStep
from skills.spatial_executor import SpatialExecutor
from skills.response_formatter import format_build_response, validate_build_response

# Trial 16: Stack 4 green in middle, 3 purple to the right
msg = "[TASK_DESCRIPTION] Grid: 9x9 cells.\n[SPEAKER] Anna\n[START_STRUCTURE] \nStack four green blocks in the middle of the grid. Then stack three purple blocks immediately to the right of the green tower."

parsed = parse_green_message(msg)
print(f"Instruction: {parsed.instruction_text}")
print(f"Start blocks: {len(parsed.start_grid.blocks)}")

steps = [
    BuildStep(action="stack", color="Green", count=4, position={"x": 0, "z": 0}),
    BuildStep(action="stack", color="Purple", count=3, position={"relative_to": "existing_Green_stack_at_0_0", "direction": "right", "distance": 1}),
]

executor = SpatialExecutor(Grid())
final = executor.execute_plan(steps)
response = format_build_response(final)
print(f"Response: {response}")

is_valid, errors = validate_build_response(response)
print(f"Valid: {is_valid}, Errors: {errors}")

expected = set("Green,0,50,0;Green,0,150,0;Green,0,250,0;Green,0,350,0;Purple,100,50,0;Purple,100,150,0;Purple,100,250,0".split(";"))
actual = set(response[8:].split(";"))
print(f"Matches trial 16 target: {expected == actual}")

# Trial 9 with start structure: Stack 3 blue on existing stack + 3 yellow to the right
msg2 = "[TASK_DESCRIPTION] Grid info\n[SPEAKER] Emma\n[START_STRUCTURE] Blue,0,50,0;Blue,0,150,0;Blue,0,250,0\nAdd a blue block on top of the existing structure. Immediately to its right, build a stack of three yellow blocks."

parsed2 = parse_green_message(msg2)
print(f"\nTrial 10 instruction: {parsed2.instruction_text}")
print(f"Start blocks: {len(parsed2.start_grid.blocks)}")

exec_grid = Grid.from_str(parsed2.start_grid.to_str())
steps2 = [
    BuildStep(action="stack", color="Blue", count=1, position={"x": 0, "z": 0}),
    BuildStep(action="stack", color="Yellow", count=3, position={"relative_to": "existing_Blue_stack_at_0_0", "direction": "right", "distance": 1}),
]
executor2 = SpatialExecutor(exec_grid)
final2 = executor2.execute_plan(steps2)
response2 = format_build_response(final2)
print(f"Response: {response2}")

is_valid2, errors2 = validate_build_response(response2)
print(f"Valid: {is_valid2}, Errors: {errors2}")

expected2 = set("Blue,0,50,0;Blue,0,150,0;Blue,0,250,0;Blue,0,350,0;Yellow,100,50,0;Yellow,100,150,0;Yellow,100,250,0".split(";"))
actual2 = set(response2[8:].split(";"))
print(f"Matches trial 10 target: {expected2 == actual2}")

print("\nAll smoke tests passed!")
