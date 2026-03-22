# AgentWhetters BWIM Purple Agent (OpenAI)

Builder agent for the AgentBeats BWIM benchmark. Uses GPT-4o-mini augmented
with a modular skills pipeline that offloads spatial computation to
deterministic code.

## Pipeline

1. **Parse.** Extract instruction text, speaker, and starting grid state from
   the green agent message.

2. **Detect underspecification.** Heuristic analysis flags missing color or
   count. If genuinely ambiguous, issue a single `[ASK]` query (color-specific
   count questions so the green agent can identify the correct stack). Priority:
   color over count. One question per round max.

3. **Structure analysis.** Analyze the current grid for geometric primitives
   (rows, stacks, L-shapes, T-shapes) and inject the description into the
   planner prompt.

4. **Plan (LLM).** GPT-4o-mini decomposes the instruction into atomic JSON
   build steps (stack, place, extend_row, etc.) using nine worked examples and
   adaptive prompt enrichment (15 pattern-matching rules that inject concept
   reminders for difficult spatial phrases).

5. **Deterministic corrections.** Four rule-based post-plan passes run before
   execution:
   - Chain reference patching (resolve "the green one" across steps)
   - Direction consistency (flip contradicted left/right/front/behind)
   - Each-end cap correction (recompute endpoints after row extension)
   - T-shape extend fix (replace invalid `on_top` with correct axis direction)

6. **Verify.** Check direction, each-expansion, stacking-vs-horizontal, and
   count plausibility. Re-plan with correction hints if critical issues found.

7. **Execute.** Deterministic engine places blocks on an in-memory grid. The
   LLM specifies only (x, z) positions; the executor computes y automatically.

8. **Format and validate.** Translate grid to `[BUILD]` protocol format.

If any stage fails, the agent falls back to a direct LLM call with a
standalone system prompt.
