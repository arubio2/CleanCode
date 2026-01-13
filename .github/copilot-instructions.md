# Copilot / AI Agent Instructions for IAthon (CleanCode)

Purpose: brief, actionable rules to get an LLM agent immediately productive in this repo.

## Quick context
- Main components:
  - `AgentResearchAssitant.py` — orchestrator: data loading, validation, ReAct loop (DecisionMaker, CodexGenerator, ReActAnalyzer), report synthesis and optional Pandoc conversion to PPTX.
  - `md_to_ppt.py` — converts a generated Markdown report + images into a styled PPTX using a template and an LLM that returns JSON slide specs.
- Headless-safe plotting is used (`matplotlib.use('Agg')`). The system expects an already-loaded `df` (no file reads inside generated code).

## How to run (examples)
- Run full analysis:
  - python AgentResearchAssitant.py -i data.csv -o output/report.md --format pptx --api-key $OPENAI_API_KEY
- Convert report -> pptx (use a template):
  - python md_to_ppt.py --md report.md --pptx my_template.pptx --api_key $OPENAI_API_KEY

## Environment & dependencies (inspect `AgentResearchAssitant.py`, `md_to_ppt.py`)
- Required (common): `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `plotly`, `python-pptx`, `openai`.
- Optional: `requests`, `beautifulsoup4` (live pricing), `Pillow` (image fits). If missing, code falls back gracefully (see FALLBACK_PRICING and checks).
- Pandoc is used when producing PPTX via conversion; ensure `pandoc` is installed when invoking `--format pptx`.

## Important safety & generation constraints (must follow)
- NEVER use file reads in generated code blocks. The `SafeRunner` prohibits `pd.read_csv()`, `pd.read_excel()` and similar — the DataFrame `df` is the input. If you need a different input, modify upstream CLI.
- Do NOT use `exec()` or similar dynamic execution in generated code. `SafeRunner` asserts against `exec(`.
- All plots must be saved into the provided `FIGURES_DIR` (a path injected into the execution environment). Do not assume working directory.
- Required file-saving pattern (example):
  - plt.savefig(os.path.join(FIGURES_DIR, 'descriptive_name_001.png'), dpi=150, bbox_inches='tight')
  - Follow with plt.close() after each save.
- Filenames should be descriptive and unique (e.g., `distribution_age.png`, `correlation_heatmap.png`, `survival_km_001.png`). Use sequential numbering when generating multiple plots.
- Generated code must be self-contained: define intermediate variables (e.g., `df_clean`) inside the block and check for column existence before use.

## LLM interfaces & expected formats (critical)
- DecisionMaker (`DecisionMaker.decide`) expects a batch plan: on step 0 it expects a "BATCH ANALYSIS: ..." description with 5–7 analyses; on later steps it expects either a targeted batch or the literal "STOP".
- CodexGenerator (`CodexGenerator.generate`) expects **pure Python code only** as output (no markdown, no backticks, no explanation). The prompt enforces the "BATCH" rules and error context.
- `synthesize_report` requires the final markdown to:
  - Use only the available image filenames (provided list) and embed them as ![desc](figures/filename.png).
  - Include a "Statistical Summary" section containing exact p-values and means.
  - End with a "Conclusions" section and **no extra content** after it.
- `md_to_ppt.py` LLM expects JSON only back with `slides` array items: `{title, bullets, image_path|null, layout_index, notes}` and must respect placeholder capacity indicators returned via `get_template_layouts()`.

## Data quality & auto-fix behavior
- `DataQualityValidator.validate()` prints issues and returns pass/fail. If it fails, the CLI interactively asks to auto-fix (y/n). Auto-fixes include capping ages and converting negative times to absolute values.
- For automation (non-interactive CI), simulate input or modify the script to skip prompts (note: interactive prompt present by design).

## Pricing / tracking
- Token usage and pricing are tracked via `UsageTracker`. It attempts live web fetch; otherwise it falls back to a cached file (`~/.cache/iathlon/openai_pricing.json`) or built-in `FALLBACK_PRICING`.
- Keep an eye on the cache file and the printed pricing source indicator: `live`, `cache`, or `fallback`.

## Debugging & smoke tests
- Basic smoke test to validate end-to-end (safe small dataset):
  1. Prepare a small CSV with a numeric column `age` and a categorical `group`.
  2. Run: `python AgentResearchAssitant.py -i small.csv -o out/report.md --api-key $OPENAI_API_KEY -v`
  3. Verify `out/figures/*.png` are created, `out/report.md` includes `figures/...` embeds, and `md_to_ppt.py` can convert it using a template.
- If PPTX save fails with PermissionError, the file might be open in PowerPoint — close it and re-run.

## Files to inspect for more context
- `AgentResearchAssitant.py` (rules for code gen, SafeRunner, DataQualityValidator, ReAct loop)
- `md_to_ppt.py` (template layout metadata, LLM slide-mapping, image placement rules)

## Small gotchas & notes
- Generated code should guard dtype conversions: use `pd.to_numeric(..., errors='coerce')` and validate existence of expected columns.
- For survival analysis include checks for `time`/`status`/`event` columns and validation of binary status columns.
- The project uses specific models in prompts (e.g., `gpt-5-mini`, `gpt-5.1-codex-mini`, `gpt-4o-mini`); prompts assume these names and behaviors.

---
If any of the above is unclear or you want examples added (short sample code blocks or a sample small CSV for smoke testing), tell me which section to expand and I will iterate. Thank you!