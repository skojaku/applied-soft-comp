# Ralph Agent Instructions

You are an autonomous agent building pedagogical marimo notebooks for Module 3 of Applied Soft Computing. Each iteration, you complete ONE user story.

## Your Task

1. **Read prd.json** - Find the next story where `passes: false`, in document order.
2. **Read progress.txt** - Check what has been built and any learnings from prior iterations.
3. **Execute acceptance criteria** - Follow each numbered step exactly.
4. **Validate** - Verify the output exists, runs, and satisfies the story.
5. **Commit** - Format: `feat: [story-id] - [short title]`
6. **Push** - Run `git push` after every commit.
7. **Update prd.json** - Set `passes: true` on the completed story.
8. **Log learnings** - Append a short note to progress.txt.

## Working Directory

All paths are relative to:
`/Users/skojaku-admin/Documents/teaching/applied-soft-comp/notebooks/applsoftcomp-agentic-coding`

## Key Constraints

- Use `litellm` for all LLM calls. Never import `openai` directly.
- Use `ChatLiteLLM` from `langchain_community.chat_models` for LangChain agents in Notebook 2.
- All notebooks must have an inline `# /// script` dependency block so they run via `uvx marimo run <file>.py`.
- The LLM config panel (model string, api_key, api_base) must be a reactive marimo state at the top of each notebook. Every cell that calls the LLM must depend on it.
- Write narrative prose in `mo.md()` cells. No bullet points. Short paragraphs (2-3 sentences). No em-dashes.
- Introduce one concept per section. Do not combine multiple ideas in one explanation block.
- Place figures before prose. Download internet figures to `figs/` and reference them locally. Generate matplotlib diagrams and save to `figs/` as well.
- Default model for testing: `ollama/glm-4.7:cloud`. Confirmed available cloud models: `glm-4.7:cloud`, `glm-5:cloud`, `qwen3.5:cloud`, `gemini-3-flash-preview:cloud`.

## File Locations

| File | Purpose |
|------|---------|
| `prd.json` | User stories — update `passes` field after each story |
| `progress.txt` | Learnings — append after each story |
| `prompt_engineering.py` | Notebook 1: Prompt Engineering |
| `react_agentic.py` | Notebook 2: ReAct and Agentic Systems |
| `figs/` | Downloaded and generated figures |

## Story Format in prd.json

Each user story has this shape:

```json
{
  "id": "US-001",
  "story": "As a student, I want ...",
  "passes": false
}
```

Set `passes: true` after you have verified the story is implemented.

## Output Conventions

### Notebook cell structure
Each concept section follows this pattern:
1. A figure (`mo.image` or inline matplotlib) that shows the concept visually.
2. One or two short `mo.md()` paragraphs explaining the concept. Two or three sentences each.
3. An interactive widget or code cell for the demo.
4. A short `mo.md()` reflection question.

### Student task callout
```python
mo.callout(
    mo.md("""
**Try it yourself**

[Task description — one short paragraph]

*Hint:* [First hint, visible by default]
"""),
    kind="info"
)
```

### progress.txt entry
```
[YYYY-MM-DD] Story US-NNN completed.
What I did: ...
Learnings: ...
Next story: US-NNN
```

## Stop Condition

When ALL stories in `prd.json` have `passes: true`, output exactly:

```
<promise>COMPLETE</promise>
```

Do not output `<promise>COMPLETE</promise>` until every story is genuinely implemented and validated.

## Rules

1. One story per iteration.
2. Follow the constraints above exactly.
3. Never overwrite existing notebook content — append new cells.
4. Always validate that `uvx marimo run <notebook>.py` would succeed (check for syntax errors and missing imports).
5. Include the model string in every litellm call so the config panel change propagates correctly.
