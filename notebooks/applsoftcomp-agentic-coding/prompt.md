# Task

You are implementing the marimo notebooks described in `prd.json`. Work through the user stories one at a time. Pick the next incomplete story, implement it, and stop. Do not implement more than one story per run.

## How to decide what to do next

1. Read `prd.json` to understand the full scope of the project.
2. Inspect the current state of the repository to see what has already been implemented.
3. Identify the lowest-numbered user story that is not yet satisfied by the existing code.
4. Implement only that story. Keep the change focused and minimal.

## Rules

- Write all notebook code as marimo `.py` files with inline `# /// script` dependency metadata so they run via `uvx marimo run <file>.py`.
- Use `litellm` for all LLM calls. Never import `openai` directly.
- Use `ChatLiteLLM` from `langchain_community.chat_models` for LangChain agents in Notebook 2.
- The LLM config panel (model string, api_key, api_base) must be defined as a reactive marimo state at the top of each notebook. All cells that call the LLM must depend on this state.
- Write narrative prose in `mo.md()` cells. No bullet points. Short paragraphs (2-3 sentences). No em-dashes.
- Introduce one concept per section. Do not combine multiple ideas in one explanation block.
- Place figures before prose. Download internet figures to `figs/` and reference them locally. Generate matplotlib diagrams and save them to `figs/` as well.
- After every write or edit, run `git add` and `git commit` with a short message describing the story implemented.
- After committing, run `git push`.

## Completion condition

After implementing and committing the story, check whether ALL user stories in `prd.json` are now satisfied by the code in the repository. If yes, output exactly:

<promise>COMPLETE</promise>

If stories remain, output a one-line summary of which story you just completed and what the next one is. Do not output `<promise>COMPLETE</promise>` until every story is genuinely implemented.
