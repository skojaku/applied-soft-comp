# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "litellm==1.82.0",
#   "pydantic==2.12.5",
#   "matplotlib==3.10.8",
#   "numpy==2.4.2",
#   "scipy==1.17.1",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prompt Engineering: The Art of Talking to LLMs

    *How phrasing shapes probability — and what you can do about it*

    /// tip | How to run this notebook
    Download this file, then open a terminal and run:

    ```
    marimo edit --sandbox prompt_engineering.py
    ```

    If you do not have marimo installed, install it first with `pip install marimo` or run it without installation using `uvx marimo edit --sandbox prompt_engineering.py`. The `--sandbox` flag creates an isolated environment and installs all dependencies automatically.
    ///

    /// note | What you'll learn in this module
    This module introduces the art and science of prompt engineering. We will explore
    how LLMs generate text as probability samplers, examine the five building blocks of
    a well-structured prompt, and understand how few-shot examples, chain-of-thought
    reasoning, and structured output constraints apply to real tasks.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # LLM Configuration Panel
    mo.md("""
    ## Configuration
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    model_input = mo.ui.text(
        value="ollama/gemma3:27b-cloud",
        label="Model (litellm format, e.g. ollama/glm-4.7:cloud, openai/gpt-4o, anthropic/claude-3-5-sonnet-20241022)",
        full_width=True,
    )
    api_key_input = mo.ui.text(
        value="",
        kind="password",
        label="API key (leave blank for local ollama)",
        full_width=True,
    )
    api_base_input = mo.ui.text(
        value="http://localhost:11434",
        label="API base URL (only needed for custom endpoints)",
        full_width=True,
    )
    config_panel = mo.vstack(
        [
            mo.md("### LLM Configuration"),
            model_input,
            api_key_input,
            api_base_input,
            mo.md(
                "*Change any field above and all cells that call the LLM will update automatically.*\n\n"
                "**Default model:** `ollama/gemma3:27b-cloud` — a free cloud model served through your local ollama installation. "
                "No API key is required. Run `ollama list` in your terminal to see all available models. "
                "To use a different local model, first run `ollama pull <model-name>` in your terminal, "
                "then update the model string above (e.g., `ollama/llama3.2`)."
            ),
        ]
    )
    config_panel
    return api_base_input, api_key_input, model_input


@app.cell(hide_code=True)
def _(mo):
    import subprocess as _subprocess

    try:
        _result = _subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        _ollama_output = _result.stdout.strip() if _result.returncode == 0 else _result.stderr.strip()
        _ollama_available = _result.returncode == 0
    except FileNotFoundError:
        _ollama_output = (
            "ollama not found. Install it from https://ollama.com or use a cloud model string (e.g. openai/gpt-4o)."
        )
        _ollama_available = False
    except Exception as _e:
        _ollama_output = f"Could not run ollama list: {_e}"
        _ollama_available = False

    if _ollama_available and _ollama_output:
        _display = mo.callout(
            mo.vstack(
                [
                    mo.md("**Available ollama models** (from `ollama list`):"),
                    mo.md(f"```\n{_ollama_output}\n```"),
                    mo.md(
                        "Use the model name above in the config panel as `ollama/<model-name>` (e.g. `ollama/llama3.2`)."
                    ),
                ]
            ),
            kind="success",
        )
    elif _ollama_available:
        _display = mo.callout(
            mo.md(
                "No local ollama models found. Run `ollama pull <model-name>` to download one, or use a cloud model string."
            ),
            kind="warn",
        )
    else:
        _display = mo.callout(
            mo.md(f"**ollama not available:** {_ollama_output}"),
            kind="warn",
        )
    mo.accordion({"Available ollama models (click to expand)": _display})
    return


@app.cell(hide_code=True)
def _(api_base_input, api_key_input, model_input):
    # Reactive config state — all downstream cells depend on these variables
    llm_model = model_input.value.strip()
    llm_api_key = api_key_input.value or None
    llm_api_base = api_base_input.value or None
    return llm_api_base, llm_api_key, llm_model


@app.cell(hide_code=True)
def _(llm_model, mo):
    import subprocess as _sp


    def _check_model_available(model_str: str) -> tuple:
        """Return (is_ok, warning_message). Warns if an ollama/local model is not in `ollama list`."""
        if not model_str.startswith("ollama/"):
            return True, ""  # non-ollama providers are not checked here
        model_name = model_str.split("ollama/", 1)[1]
        try:
            _r = _sp.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            if _r.returncode != 0:
                return True, ""  # can't check — assume OK
            installed = [line.split()[0] for line in _r.stdout.strip().splitlines()[1:] if line.strip()]
            if not any(model_name == m or model_name in m for m in installed):
                return False, (
                    f"Model `{model_name}` was not found in your local ollama registry.\n\n"
                    f"**To fix this, run one of the following in your terminal:**\n\n"
                    f"```\nollama pull {model_name}\n```\n\n"
                    f"Or switch to a cloud model string (e.g. `ollama/glm-4.7:cloud`, `openai/gpt-4o`) "
                    f"in the configuration panel above.\n\n"
                    f"**Currently installed models:** {', '.join(installed) if installed else 'none'}"
                )
        except FileNotFoundError:
            pass  # ollama not installed — not a local model issue
        except Exception:
            pass
        return True, ""


    _model_ok, _model_warning = _check_model_available(llm_model)
    if not _model_ok:
        mo.callout(
            mo.md(f"**Model not found — {_model_warning}"),
            kind="danger",
        )
    else:
        mo.md(f"*Model `{llm_model}` looks available.*")
    return


@app.cell(hide_code=True)
def _(llm_api_base, llm_api_key, llm_model, mo):
    import litellm


    def call_llm(prompt: str, system: str = "") -> str:
        """Call the configured LLM and return the response text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            kwargs = {"model": llm_model, "messages": messages}
            if llm_api_key:
                kwargs["api_key"] = llm_api_key
            if llm_api_base:
                kwargs["api_base"] = llm_api_base
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower() or "model" in error_msg.lower():
                return (
                    f"**Error:** Model `{llm_model}` was not found. "
                    "Please run `ollama pull <model>` to download it locally, "
                    "or switch to a cloud model in the configuration panel above."
                )
            return f"**Error calling LLM:** {error_msg}"


    mo.accordion(
        {
            "call_llm() helper — click to view": mo.md(
                f"*Using model: `{llm_model}`*\n\n"
                "```python\n"
                "def call_llm(prompt: str, system: str = '') -> str:\n"
                "    # Calls the configured LLM via litellm.completion()\n"
                "    # Handles api_key, api_base, and friendly error messages\n"
                "    ...\n"
                "```"
            )
        }
    )
    return call_llm, litellm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 1: Why Prompts Matter — LLMs as Probability Samplers

    Every LLM response is a sample from a probability distribution. The model does not
    retrieve a fixed answer stored somewhere in its weights. It generates the most likely
    continuation of your exact input, token by token. Change a single word, and you shift
    the distribution. Change the structure of your request, and you can shift it dramatically.

    Let us see this in action. Below are two prompts for the same task. One is vague; the
    other is precise. Run both and compare what you get.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    vague_prompt = mo.ui.text_area(
        value="Tell me about climate change.",
        label="Vague prompt",
        full_width=True,
        rows=3,
    )
    precise_prompt = mo.ui.text_area(
        value=(
            "You are an environmental scientist writing for a general audience. "
            "Summarize the three most important human-caused drivers of climate change "
            "in exactly three sentences, one sentence per driver."
        ),
        label="Precise prompt",
        full_width=True,
        rows=4,
    )
    run_comparison_btn = mo.ui.run_button(label="Run both prompts")
    mo.vstack([vague_prompt, precise_prompt, run_comparison_btn])
    return precise_prompt, run_comparison_btn, vague_prompt


@app.cell(hide_code=True)
def _(call_llm, mo, precise_prompt, run_comparison_btn, vague_prompt):
    if run_comparison_btn.value:
        vague_response = call_llm(vague_prompt.value)
        precise_response = call_llm(precise_prompt.value)
        _output = mo.vstack(
            [
                mo.md("### Vague prompt response"),
                mo.callout(mo.md(vague_response), kind="warn"),
                mo.md("### Precise prompt response"),
                mo.callout(mo.md(precise_response), kind="success"),
                mo.md(
                    "*Reflection: What was different about the two responses? "
                    "Which was more useful, and why did phrasing produce that difference?*"
                ),
            ]
        )
    else:
        _output = mo.md("*Click **Run both prompts** to see the comparison.*")
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 2: The Five Building Blocks of a Prompt

    A well-structured prompt is not just a question. It has up to five components, each
    doing a different job. Let us meet them one at a time.

    **Instruction** tells the model what task to perform. Without it, the model guesses.
    A clear instruction anchors the response to the right task type.

    **Data** is the material the model should work on. Separating data from the instruction
    keeps the prompt readable and makes it easy to swap the data without rewriting everything else.

    **Format** constrains the shape of the output. Asking for a numbered list, a JSON object,
    or exactly three sentences each changes what the model generates, because format constraints
    shift the probability mass toward tokens that fit the required structure.

    **Persona** assigns a role to the model. Telling the model it is a pediatric nurse versus a
    software engineer versus a poet changes its vocabulary, tone, and depth of technical detail.

    **Context** provides background the model needs but does not have from the instruction alone.
    A date, a domain, a prior conversation, or a constraint about the audience all belong here.

    Now toggle each component on and off. Watch how the assembled prompt changes in real time,
    then send it and compare the responses.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    use_instruction = mo.ui.checkbox(value=True, label="Instruction")
    use_data = mo.ui.checkbox(value=False, label="Data")
    use_format = mo.ui.checkbox(value=False, label="Format")
    use_persona = mo.ui.checkbox(value=False, label="Persona")
    use_context = mo.ui.checkbox(value=False, label="Context")

    instruction_text = mo.ui.text_area(
        value="Classify the sentiment of the following review as Positive, Negative, or Neutral.",
        label="Instruction text",
        full_width=True,
        rows=2,
    )
    data_text = mo.ui.text_area(
        value='Review: "The battery lasts all day and the screen is gorgeous, but the price is absurd."',
        label="Data text",
        full_width=True,
        rows=2,
    )
    format_text = mo.ui.text_area(
        value="Respond with exactly one word: Positive, Negative, or Neutral.",
        label="Format text",
        full_width=True,
        rows=2,
    )
    persona_text = mo.ui.text_area(
        value="You are a professional product reviewer with 10 years of experience in consumer electronics.",
        label="Persona text",
        full_width=True,
        rows=2,
    )
    context_text = mo.ui.text_area(
        value="This review was posted on a budget-focused tech forum where readers care primarily about value for money.",
        label="Context text",
        full_width=True,
        rows=2,
    )

    run_blocks_btn = mo.ui.run_button(label="Send assembled prompt")

    mo.vstack(
        [
            mo.md("### Toggle components"),
            mo.hstack([use_instruction, use_data, use_format, use_persona, use_context]),
            mo.md("### Edit component text"),
            instruction_text,
            data_text,
            format_text,
            persona_text,
            context_text,
            run_blocks_btn,
        ]
    )
    return (
        context_text,
        data_text,
        format_text,
        instruction_text,
        persona_text,
        run_blocks_btn,
        use_context,
        use_data,
        use_format,
        use_instruction,
        use_persona,
    )


@app.cell(hide_code=True)
def _(
    call_llm,
    context_text,
    data_text,
    format_text,
    instruction_text,
    mo,
    persona_text,
    run_blocks_btn,
    use_context,
    use_data,
    use_format,
    use_instruction,
    use_persona,
):
    parts = []
    if use_persona.value:
        parts.append(persona_text.value)
    if use_context.value:
        parts.append(f"Context: {context_text.value}")
    if use_instruction.value:
        parts.append(instruction_text.value)
    if use_data.value:
        parts.append(data_text.value)
    if use_format.value:
        parts.append(f"Format: {format_text.value}")

    assembled = "\n\n".join(parts) if parts else "(No components selected — toggle at least one above.)"

    _preview = mo.vstack(
        [
            mo.md("### Assembled prompt (live preview)"),
            mo.callout(mo.md(f"```\n{assembled}\n```"), kind="info"),
            mo.md("*This preview updates in real time as you toggle components above.*"),
        ]
    )

    blocks_response = None
    if run_blocks_btn.value and parts:
        blocks_response = call_llm(assembled)
        _display = mo.vstack(
            [
                _preview,
                mo.md("### Model response"),
                mo.callout(mo.md(blocks_response), kind="success"),
                mo.md("*Reflection: Which component changed the response the most when you toggled it?*"),
            ]
        )
    else:
        _display = _preview
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 3: Few-Shot Learning — Examples as Activators

    When you show a model an example of the task alongside the instruction, you activate
    a much more specific pattern in its probability distribution. Two or three examples
    are usually enough. More than five rarely helps.

    The demo below uses sentiment classification. Use the slider to choose how many
    examples to include (0, 1, 2, or 5), then run the model. A table records each
    configuration so you can compare across runs.

    After trying several slider positions, hit **Shuffle examples and run** to randomize
    the order of the same examples and run again. Watch whether the prediction changes.
    This is order bias: the most recently seen label often has outsized influence on
    what the model predicts next.

    /// warning | Few-shot prompts are brittle in production
    Small changes in example order can flip the output entirely. A few-shot prompt that
    works reliably in testing may behave differently in production when examples are
    reordered or replaced. Always test multiple orderings before deploying.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    SENTIMENT_EXAMPLES = [
        ("The food was incredible and the service was friendly.", "Positive"),
        ("I waited 45 minutes and my order was completely wrong.", "Negative"),
        ("The product works as described, nothing special.", "Neutral"),
        ("Absolutely love this! Best purchase I've made all year.", "Positive"),
        ("Broke after two days. Total waste of money.", "Negative"),
    ]

    fewshot_slider = mo.ui.slider(
        steps=[0, 1, 2, 5],
        value=0,
        label="Number of few-shot examples (0 = zero-shot, 1 = one-shot, 2 = two-shot, 5 = five-shot)",
    )
    shuffle_btn = mo.ui.run_button(label="Shuffle examples and run")
    run_fewshot_btn = mo.ui.run_button(label="Run with current settings")
    with_replacement_toggle = mo.ui.switch(label="Sample with replacement", value=False)

    fewshot_test = mo.ui.text_area(
        value="The app crashes every time I open it, but customer support was surprisingly helpful.",
        label="Test sentence (classify this)",
        full_width=True,
        rows=2,
    )
    return (
        SENTIMENT_EXAMPLES,
        fewshot_slider,
        fewshot_test,
        run_fewshot_btn,
        shuffle_btn,
        with_replacement_toggle,
    )


@app.cell(hide_code=True)
def _(
    SENTIMENT_EXAMPLES,
    fewshot_slider,
    fewshot_test,
    mo,
    run_fewshot_btn,
    shuffle_btn,
    with_replacement_toggle,
):
    n_selected = fewshot_slider.value
    use_replacement = with_replacement_toggle.value
    _example_rows = []
    for _i, (_text, _label) in enumerate(SENTIMENT_EXAMPLES):
        if use_replacement and n_selected > 0:
            _example_rows.append(f"| 🎲 {_i + 1} | {_text} | {_label} |")
        elif _i < n_selected:
            _example_rows.append(f"| ✅ **{_i + 1}** | {_text} | **{_label}** |")
        else:
            _example_rows.append(f"| ⬜ {_i + 1} | {_text} | {_label} |")
    _pool_note = (
        "**Example pool** (🎲 = all in pool, sampled randomly with replacement)"
        if use_replacement and n_selected > 0
        else "**Example pool** (✅ = fed into the prompt, ⬜ = excluded)"
    )
    _examples_table = mo.md(_pool_note + "\n\n| # | Text | Label |\n|---|------|-------|\n" + "\n".join(_example_rows))
    mo.vstack(
        [
            fewshot_slider,
            with_replacement_toggle,
            _examples_table,
            fewshot_test,
            mo.hstack([run_fewshot_btn, shuffle_btn]),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    import random

    get_fewshot_results, set_fewshot_results = mo.state([])
    return get_fewshot_results, random, set_fewshot_results


@app.cell(hide_code=True)
def _(
    SENTIMENT_EXAMPLES,
    call_llm,
    fewshot_slider,
    fewshot_test,
    get_fewshot_results,
    mo,
    random,
    run_fewshot_btn,
    set_fewshot_results,
    shuffle_btn,
    with_replacement_toggle,
):
    def _build_fewshot_prompt(examples, n, test_sentence, shuffled=False, with_replacement=False):
        if n == 0:
            selected = []
        elif with_replacement:
            selected = random.choices(examples, k=n)
        else:
            selected = list(examples[:n])
        if shuffled and n > 0:
            random.shuffle(selected)
        order_str = " → ".join(label for _, label in selected) if selected else "none"
        prompt_parts = ["Classify the sentiment as Positive, Negative, or Neutral.\n"]
        for text, label in selected:
            prompt_parts.append(f'Text: "{text}"\nSentiment: {label}\n')
        prompt_parts.append(f'Text: "{test_sentence}"\nSentiment:')
        return "\n".join(prompt_parts), order_str


    _output = None
    if run_fewshot_btn.value or shuffle_btn.value:
        is_shuffled = bool(shuffle_btn.value)
        is_replacement = bool(with_replacement_toggle.value)
        n = fewshot_slider.value
        if is_shuffled and n == 0:
            _output = mo.callout(
                mo.md("**Note:** Shuffling has no effect in zero-shot mode. Set the slider to at least 1."),
                kind="warn",
            )
        else:
            _prompt, _order = _build_fewshot_prompt(SENTIMENT_EXAMPLES, n, fewshot_test.value, is_shuffled, is_replacement)
            _resp = call_llm(_prompt)
            _prediction = _resp.strip().split("\n")[0]
            _current = get_fewshot_results()
            set_fewshot_results(
                _current
                + [
                    {
                        "n_examples": n,
                        "shuffled": "Yes" if is_shuffled else "No",
                        "replacement": "Yes" if is_replacement else "No",
                        "order": _order,
                        "prediction": _prediction,
                    }
                ]
            )

    _rows = get_fewshot_results()
    if _output is None:
        if _rows:
            _table_data = {
                "Examples": [r["n_examples"] for r in _rows],
                "Shuffled": [r["shuffled"] for r in _rows],
                "With replacement": [r.get("replacement", "No") for r in _rows],
                "Example order (labels)": [r["order"] for r in _rows],
                "Prediction": [r["prediction"] for r in _rows],
            }
            _output = mo.vstack(
                [
                    mo.md("### Results table"),
                    mo.ui.table(_table_data),
                    mo.md(
                        "*Reflection: Did the order of examples change the prediction? "
                        "What does this tell you about using few-shot prompts in production?*"
                    ),
                ]
            )
        else:
            _output = mo.md("*Run the model to see results appear here.*")
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 4: Chain-of-Thought — Thinking Out Loud

    Sometimes asking the model to think out loud, step by step, leads to better answers.
    This technique is called chain-of-thought prompting.

    The puzzle below requires two or more intermediate steps to solve correctly. Click
    **Run both modes** to run the puzzle twice: first in direct-answer mode to establish
    a baseline, then in chain-of-thought mode so you can compare the two responses side by side.

    One important caveat: the reasoning steps can sound completely plausible but still lead
    to a wrong answer. The final answer may even appear before the reasoning is finished.
    Chain-of-thought is a tool, not a guarantee.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    COT_PUZZLE = (
        "A farmer has 17 sheep. 9 die. How many sheep does the farmer have left? "
        "Now, if he sells half of the remaining sheep and buys 3 more, how many does he have?"
    )

    run_direct_btn = mo.ui.run_button(label="Run direct-answer (baseline)")
    run_cot_btn = mo.ui.run_button(label="Run chain-of-thought")
    run_both_btn = mo.ui.run_button(label="Run both modes")

    mo.vstack(
        [
            mo.callout(mo.md(f"**Puzzle:** {COT_PUZZLE}"), kind="info"),
            mo.hstack([run_direct_btn, run_cot_btn, run_both_btn]),
            mo.md("*Use **Run both modes** for a side-by-side comparison, or run each mode individually.*"),
        ]
    )
    return COT_PUZZLE, run_both_btn, run_cot_btn, run_direct_btn


@app.cell(hide_code=True)
def _(COT_PUZZLE, call_llm, mo, run_both_btn, run_cot_btn, run_direct_btn):
    _direct_sys = "Answer directly and concisely."
    _cot_sys = "Think through this step by step before giving your final answer. Show all intermediate reasoning."

    if run_both_btn.value:
        _direct_resp = call_llm(COT_PUZZLE, system=_direct_sys)
        _cot_resp = call_llm(COT_PUZZLE, system=_cot_sys)
        _output = mo.vstack(
            [
                mo.md("### Side-by-side comparison"),
                mo.hstack(
                    [
                        mo.vstack(
                            [
                                mo.md("**Direct-answer (baseline)**"),
                                mo.callout(mo.md(_direct_resp), kind="warn"),
                            ]
                        ),
                        mo.vstack(
                            [
                                mo.md("**Chain-of-thought**"),
                                mo.callout(mo.md(_cot_resp), kind="success"),
                            ]
                        ),
                    ]
                ),
                mo.md(
                    "*Reflection: Did the reasoning steps actually lead to the correct answer? "
                    "Did chain-of-thought help, hurt, or make no difference here?*"
                ),
            ]
        )
    elif run_direct_btn.value:
        _direct_resp = call_llm(COT_PUZZLE, system=_direct_sys)
        _output = mo.vstack(
            [
                mo.md("### Response (direct-answer baseline)"),
                mo.callout(mo.md(_direct_resp), kind="warn"),
                mo.md("*This is your baseline. Now click **Run chain-of-thought** to compare.*"),
            ]
        )
    elif run_cot_btn.value:
        _cot_resp = call_llm(COT_PUZZLE, system=_cot_sys)
        _output = mo.vstack(
            [
                mo.md("### Response (chain-of-thought)"),
                mo.callout(mo.md(_cot_resp), kind="success"),
                mo.md(
                    "*Reflection: Did the reasoning steps actually lead to the correct answer? "
                    "Did chain-of-thought help, hurt, or make no difference here?*"
                ),
            ]
        )
    else:
        _output = mo.md(
            "*Click **Run direct-answer (baseline)**, **Run chain-of-thought**, or **Run both modes** to see responses.*"
        )
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 5: Structured Output — JSON Schema Constraints

    Asking the model to "respond in JSON" in plain text does not guarantee valid JSON.
    The model is still sampling tokens freely and can produce a malformed structure.

    The alternative is to pass a Pydantic schema as a `response_format` constraint.
    This works at the token level: the model cannot generate a response that violates
    the schema. Toggle between the two approaches below and count how often each produces
    valid output.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    INVOICE_TEXT = (
        "Invoice #4821 — Billed to: Acme Corp. "
        "Date: March 15, 2025. "
        "Services rendered: Strategic consulting (40 hrs @ $150/hr). "
        "Total amount due: $6,000.00."
    )

    structured_toggle = mo.ui.switch(label="Use Pydantic schema constraint", value=False)
    run_structured_btn = mo.ui.run_button(label="Extract invoice data")

    mo.vstack(
        [
            mo.callout(mo.md(f"**Invoice text:**\n\n{INVOICE_TEXT}"), kind="info"),
            structured_toggle,
            run_structured_btn,
        ]
    )
    return INVOICE_TEXT, run_structured_btn, structured_toggle


@app.cell
def _():
    import json
    from pydantic import BaseModel


    # The schema below is the contract passed to the LLM as response_format.
    # The model cannot generate output that violates these fields.
    class InvoiceData(BaseModel):
        client_name: str
        date: str
        total_amount: str

    return InvoiceData, json


@app.cell(hide_code=True)
def _(
    INVOICE_TEXT,
    InvoiceData,
    call_llm,
    json,
    litellm,
    llm_api_base,
    llm_api_key,
    llm_model,
    mo,
    run_structured_btn,
    structured_toggle,
):
    if run_structured_btn.value:
        if structured_toggle.value:
            try:
                _kwargs = {
                    "model": llm_model,
                    "response_format": InvoiceData,
                    "messages": [
                        {"role": "user", "content": f"Extract client name, date, and total amount:\n\n{INVOICE_TEXT}"}
                    ],
                }
                if llm_api_key:
                    _kwargs["api_key"] = llm_api_key
                if llm_api_base:
                    _kwargs["api_base"] = llm_api_base
                _parsed = InvoiceData.model_validate_json(litellm.completion(**_kwargs).choices[0].message.content)
                _output = mo.vstack(
                    [
                        mo.md("### Structured output (schema-constrained)"),
                        mo.callout(mo.md(f"```json\n{_parsed.model_dump_json(indent=2)}\n```"), kind="success"),
                        mo.md("**Valid JSON: ✅**"),
                    ]
                )
            except Exception as _e:
                _output = mo.callout(mo.md(f"**Error:** {_e}"), kind="danger")
        else:
            _resp_text = call_llm(f"Extract client name, date, and total amount in JSON:\n\n{INVOICE_TEXT}")
            try:
                _s = _resp_text.find("{")
                _e2 = _resp_text.rfind("}") + 1
                json.loads(_resp_text[_s:_e2] if _s >= 0 else _resp_text)
                _valid, _kind = "✅", "success"
            except Exception:
                _valid, _kind = "❌", "danger"
            _output = mo.vstack(
                [
                    mo.md("### Free-text JSON response"),
                    mo.callout(mo.md(f"```\n{_resp_text}\n```"), kind="info"),
                    mo.md(f"**Valid JSON: {_valid}**"),
                ]
            )
    else:
        _output = mo.md("*Click **Extract invoice data** to run the demo.*")
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 6: Hallucination and Uncertainty Permission

    LLMs are trained to be helpful. This means they often produce a confident-sounding
    answer even when they have no reliable information. This is hallucination.

    One simple intervention is uncertainty permission: telling the model it is acceptable
    to say "I do not know." Toggle this below and ask the model about a fabricated paper.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    HALLUCINATION_QUERY = (
        "Summarize the key findings of the 2019 paper by Dr. Elara Voss titled "
        "'Quantum Resonance Patterns in Migratory Bird Navigation: A Unified Framework.' "
        "What were the main conclusions and how were they validated?"
    )

    uncertainty_toggle = mo.ui.switch(label="Allow 'I don't know' response", value=False)
    run_hallucination_btn = mo.ui.run_button(label="Ask about the paper")

    mo.vstack(
        [
            mo.callout(mo.md(f"**Question:** {HALLUCINATION_QUERY}"), kind="info"),
            uncertainty_toggle,
            run_hallucination_btn,
        ]
    )
    return HALLUCINATION_QUERY, run_hallucination_btn, uncertainty_toggle


@app.cell(hide_code=True)
def _(
    HALLUCINATION_QUERY,
    call_llm,
    mo,
    run_hallucination_btn,
    uncertainty_toggle,
):
    if run_hallucination_btn.value:
        if uncertainty_toggle.value:
            _sys = "You are a helpful assistant. If you are not certain about something, say so clearly. It is perfectly acceptable to say 'I do not know.'"
        else:
            _sys = "You are a helpful assistant."
        _h_resp = call_llm(HALLUCINATION_QUERY, system=_sys)
        _output = mo.vstack(
            [
                mo.md(
                    f"### Response ({'uncertainty permitted' if uncertainty_toggle.value else 'no uncertainty permission'})"
                ),
                mo.callout(mo.md(_h_resp), kind="success" if uncertainty_toggle.value else "warn"),
                mo.md(
                    "*Reflection: Did the uncertainty permission change the model's confidence? "
                    "Should it always be included in production prompts?*"
                ),
            ]
        )
    else:
        _output = mo.md("*Click **Ask about the paper** to see what happens.*")
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## Student Task 1: Gaussian by Prompt"),
            mo.callout(
                mo.md(
                    r"""
    **Try it yourself**

    Your goal is to craft a prompt that makes the LLM output 200 numbers following a
    standard normal distribution N(0,1). Write your prompt in the editor below and click
    **Run**. The notebook will extract numbers from the response automatically and plot a
    histogram overlaid with the true Gaussian PDF. A KS-test p-value tells you whether
    your prompt succeeded.

    **Extension:** Try changing the target distribution to N(2, 0.5). How does your prompt
    need to change?
                """
                ),
                kind="info",
            ),
            mo.callout(
                mo.md(
                    "**Hint:** The LLM has no random number generator. It approximates from its "
                    "training distribution. Be extremely specific about the count: ask for exactly "
                    "200 numbers and output only the numbers, nothing else."
                ),
                kind="neutral",
            ),
            mo.accordion(
                {
                    "Show more (detailed hint)": mo.md(
                        "For N(0,1) with 200 samples, tell the model exactly how many numbers should "
                        "fall in each bin. Roughly 68 numbers should fall between -1 and 1, about 27 "
                        "between 1 and 2 (and symmetrically -2 to -1), and only about 5 beyond ±2. "
                        "The more precisely you specify the bin counts, the more the output looks "
                        "Gaussian. You can even list expected counts: '[-3,-2]: 1, [-2,-1]: 5, "
                        "[-1,0]: 34, [0,1]: 34, [1,2]: 5, [2,3]: 1' and ask the model to match them."
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    gaussian_prompt = mo.ui.text_area(
        value=(
            "Generate exactly 200 numbers that follow a standard normal distribution N(0,1). "
            "Output only the numbers separated by spaces, nothing else."
        ),
        label="Your prompt",
        full_width=True,
        rows=5,
    )
    run_gaussian_btn = mo.ui.run_button(label="Run and evaluate")
    mo.vstack([gaussian_prompt, run_gaussian_btn])
    return gaussian_prompt, run_gaussian_btn


@app.cell(hide_code=True)
def _(mo):
    import re
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt


    def extract_numbers(text: str) -> list:
        """Extract all floating-point numbers from a string."""
        pattern = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
        return [float(m) for m in re.findall(pattern, text)]


    mo.accordion(
        {
            "Helper: extract_numbers() — click to see source": mo.md(
                "```python\n"
                "def extract_numbers(text: str) -> list:\n"
                '    """Extract all floating-point numbers from a string."""\n'
                '    pattern = r"-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?"\n'
                "    return [float(m) for m in re.findall(pattern, text)]\n"
                "```"
            ),
        }
    )
    return extract_numbers, np, plt, stats


@app.cell(hide_code=True)
def _(
    call_llm,
    extract_numbers,
    gaussian_prompt,
    mo,
    np,
    plt,
    run_gaussian_btn,
    stats,
):
    _gauss_display = mo.md("*Write your prompt above and click **Run and evaluate** to see the histogram.*")
    if run_gaussian_btn.value:
        _g_resp = call_llm(gaussian_prompt.value)
        _numbers = extract_numbers(_g_resp)

        if len(_numbers) < 10:
            _gauss_display = mo.callout(
                mo.md(
                    f"Only {len(_numbers)} numbers were extracted. Try refining your prompt to produce more numeric output."
                ),
                kind="warn",
            )
        else:
            _ks_stat, _ks_p = stats.kstest(_numbers, "norm")
            _passed = _ks_p > 0.05

            _fig, _ax = plt.subplots(figsize=(8, 4))
            _ax.hist(_numbers, bins=30, density=True, alpha=0.6, label=f"LLM output (n={len(_numbers)})")
            _x = np.linspace(-4, 4, 200)
            _ax.plot(_x, stats.norm.pdf(_x), "r-", lw=2, label="True N(0,1) PDF")
            _ax.set_xlabel("Value")
            _ax.set_ylabel("Density")
            _ax.set_title("LLM-generated numbers vs. N(0,1)")
            _ax.legend()
            plt.tight_layout()

            _indicator = "✅ PASS" if _passed else "❌ FAIL"
            _kind = "success" if _passed else "danger"

            _gauss_display = mo.vstack(
                [
                    mo.as_html(_fig),
                    mo.callout(
                        mo.md(
                            f"**KS-test p-value:** {_ks_p:.4f} — **{_indicator}**\n\n"
                            f"Extracted {len(_numbers)} numbers. "
                            f"{'The distribution matches N(0,1) at the 5% significance level.' if _passed else 'The distribution does not match N(0,1). Refine your prompt and try again.'}"
                        ),
                        kind=_kind,
                    ),
                ]
            )
    _gauss_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## Student Task 2: When Does Chain-of-Thought Fail?"),
            mo.callout(
                mo.md(
                    r"""
    **Try it yourself**

    Three puzzles are provided below, ranging from trivial arithmetic to a harder logic
    problem. Run each puzzle twice: once in direct-answer mode and once with chain-of-thought.
    The notebook records results in a table. After filling the table, write one short paragraph
    in the text area: for which puzzle type did CoT help, and when did it produce a wrong answer
    with convincing-looking reasoning?
                """
                ),
                kind="info",
            ),
            mo.callout(
                mo.md(
                    "**Hint:** Pay attention to whether the intermediate reasoning steps are "
                    "actually used to reach the final answer, or whether the final answer "
                    "appears before the reasoning is complete."
                ),
                kind="neutral",
            ),
            mo.accordion(
                {
                    "Show more (detailed hint)": mo.md(
                        "Chain-of-thought helps most on multi-step problems where each step builds "
                        "on the last, such as the logic puzzle. It tends to hurt on simple arithmetic "
                        "where the model already 'knows' the answer and the extra steps just add "
                        "opportunities for error. Watch for the model writing a final answer at the "
                        "top of its response and then 'reasoning' backward to justify it. That is a "
                        "sign that CoT is decorative, not causal."
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    COT_PUZZLES = [
        {
            "id": "arithmetic",
            "question": "What is 17 × 23?",
            "expected": "391",
        },
        {
            "id": "logic",
            "question": (
                "There are three boxes. One contains only apples, one contains only oranges, "
                "and one contains both apples and oranges. All three boxes are mislabeled. "
                "You may pick one fruit from one box. Which box do you pick from to correctly "
                "label all three boxes?"
            ),
            "expected": "The box labeled 'Apples and Oranges'",
        },
        {
            "id": "combinatorics",
            "question": (
                "A password must be exactly 4 characters long. Each character is either a digit (0-9) "
                "or an uppercase letter (A-Z). How many passwords are possible if the first character "
                "must be a letter and no character may repeat?"
            ),
            "expected": "26 × 35 × 34 × 33 = 1,021,020",
        },
    ]

    _puzzle_options = {p["question"][:60] + "...": p["id"] for p in COT_PUZZLES}
    puzzle_selector = mo.ui.dropdown(
        options=_puzzle_options,
        value=next(iter(_puzzle_options)),
        label="Select puzzle",
    )
    cot2_toggle = mo.ui.switch(label="Enable chain-of-thought", value=False)
    run_cot2_btn = mo.ui.run_button(label="Run selected puzzle")

    mo.vstack([puzzle_selector, cot2_toggle, run_cot2_btn])
    return COT_PUZZLES, cot2_toggle, puzzle_selector, run_cot2_btn


@app.cell(hide_code=True)
def _(mo):
    get_cot2_results, set_cot2_results = mo.state([])
    return get_cot2_results, set_cot2_results


@app.cell(hide_code=True)
def _(
    COT_PUZZLES,
    call_llm,
    cot2_toggle,
    get_cot2_results,
    mo,
    puzzle_selector,
    run_cot2_btn,
    set_cot2_results,
):
    if run_cot2_btn.value:
        _puzzle = next(p for p in COT_PUZZLES if p["id"] == puzzle_selector.value)
        if cot2_toggle.value:
            _sys2 = "Think step by step before giving your final answer. Show all intermediate reasoning steps."
            _mode2 = "Chain-of-thought"
        else:
            _sys2 = "Answer directly and concisely. Give only the final answer."
            _mode2 = "Direct"
        _resp2 = call_llm(_puzzle["question"], system=_sys2)
        _expected_lower = _puzzle["expected"].lower()
        _resp_lower = _resp2.lower()
        # Check if key digits or words from the expected answer appear in the response
        _correct = any(token in _resp_lower for token in _expected_lower.replace(",", "").split()[:3])
        _current2 = get_cot2_results()
        set_cot2_results(
            _current2
            + [
                {
                    "puzzle": _puzzle["id"],
                    "mode": _mode2,
                    "expected": _puzzle["expected"],
                    "response": _resp2[:100] + "..." if len(_resp2) > 100 else _resp2,
                    "correct": "✅" if _correct else "❌",
                }
            ]
        )

    _rows2 = get_cot2_results()
    if _rows2:
        _output = mo.vstack(
            [
                mo.ui.table(
                    {
                        "Puzzle": [r["puzzle"] for r in _rows2],
                        "Mode": [r["mode"] for r in _rows2],
                        "Expected": [r["expected"] for r in _rows2],
                        "Response (truncated)": [r["response"] for r in _rows2],
                        "Correct?": [r["correct"] for r in _rows2],
                    }
                ),
                mo.md(
                    "*Tip: Run each puzzle in both Direct and Chain-of-thought mode to compare. "
                    "Look at the Correct? column to see where CoT helped and where it hurt.*"
                ),
            ]
        )
    else:
        _output = mo.md("*Select a puzzle, choose a mode, and click **Run** to add rows to the table.*")
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    reflection_cot = mo.ui.text_area(
        label="Your reflection: For which puzzle type did CoT help? When did it produce wrong but convincing reasoning?",
        full_width=True,
        rows=5,
    )
    reflection_cot
    return


@app.cell(hide_code=True)
def _(COT_PUZZLES, get_cot2_results, mo):
    # Success criterion: student must run both Direct and Chain-of-thought for all 3 puzzles.
    _results = get_cot2_results()
    _puzzle_ids = {p["id"] for p in COT_PUZZLES}
    _direct_puzzles = {r["puzzle"] for r in _results if r["mode"] == "Direct"}
    _cot_puzzles = {r["puzzle"] for r in _results if r["mode"] == "Chain-of-thought"}
    _covered_direct = _direct_puzzles >= _puzzle_ids
    _covered_cot = _cot_puzzles >= _puzzle_ids
    _correct_any = any(r["correct"] == "✅" for r in _results)

    if _covered_direct and _covered_cot:
        _status = "success"
        _msg = (
            "**✅ Task 2 complete.** You have run both Direct and Chain-of-thought modes "
            "across all three puzzles. Check the Correct? column to identify where CoT helped "
            "and where it failed."
        )
    elif _results:
        _missing_d = _puzzle_ids - _direct_puzzles
        _missing_c = _puzzle_ids - _cot_puzzles
        _lines = ["**Progress so far:**"]
        if _missing_d:
            _lines.append(f"Still need Direct mode for: {', '.join(_missing_d)}")
        if _missing_c:
            _lines.append(f"Still need Chain-of-thought mode for: {', '.join(_missing_c)}")
        _msg = "\n\n".join(_lines)
        _status = "warn"
    else:
        _msg = "**No runs yet.** Select a puzzle, choose a mode, and click **Run** to begin."
        _status = "neutral"

    _output = mo.callout(mo.md(_msg), kind=_status)
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## Student Task 3: Few-Shot Order Sensitivity"),
            mo.callout(
                mo.md(
                    r"""
    **Try it yourself**

    Four labeled sentiment examples are provided below. The notebook will run the model 10
    times with the same examples but in a different random order each time, recording the
    prediction for a fixed test sentence. A bar chart shows how often each class was predicted.
    Compare this distribution to a zero-shot baseline. What does the variance tell you about
    using few-shot prompts in a real application?
                """
                ),
                kind="info",
            ),
            mo.callout(
                mo.md(
                    "**Hint:** The most recently seen example label often has outsized influence "
                    "on the model's prediction. This is called the recency bias in few-shot prompting."
                ),
                kind="neutral",
            ),
            mo.accordion(
                {
                    "Show more (detailed hint)": mo.md(
                        "To see recency bias clearly, look at which label appears last in each shuffled "
                        "ordering and check whether the prediction matches it. If you see a strong "
                        "correlation, the model is not reasoning about all examples equally. It is "
                        "weighting the most recent label disproportionately. In production, this means "
                        "the order in which you write your examples can matter as much as their content."
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    FEWSHOT_BIAS_EXAMPLES = [
        ("The concert was unforgettable. Every song was perfect.", "Positive"),
        ("The delivery was late and the packaging was damaged.", "Negative"),
        ("It is an average product. Does what it says on the box.", "Neutral"),
        ("I have never been so disappointed. Complete failure.", "Negative"),
        ("Five stars. I would recommend this to everyone I know.", "Positive"),
        ("Outstanding quality. Far exceeded my expectations.", "Positive"),
    ]

    FEWSHOT_BIAS_TEST = "The interface is clean but the performance could be better."

    run_bias_btn = mo.ui.run_button(label="Run 10 shuffled trials")

    _example_rows = "\n".join(
        f"| {_i+1} | {_text} | {_label} |"
        for _i, (_text, _label) in enumerate(FEWSHOT_BIAS_EXAMPLES)
    )
    _examples_table = mo.md(
        "**Example pool** (all four are shuffled into a random order for each trial)\n\n"
        "| # | Text | Label |\n"
        "|---|------|-------|\n"
        + _example_rows
    )

    mo.vstack(
        [
            _examples_table,
            mo.md(f"**Test sentence:** *{FEWSHOT_BIAS_TEST}*"),
            mo.md("Click the button to run 10 trials with shuffled example orders."),
            run_bias_btn,
        ]
    )
    return FEWSHOT_BIAS_EXAMPLES, FEWSHOT_BIAS_TEST, run_bias_btn


@app.cell(hide_code=True)
def _(mo):
    get_bias_predictions, set_bias_predictions = mo.state([])
    return get_bias_predictions, set_bias_predictions


@app.cell(hide_code=True)
def _(
    FEWSHOT_BIAS_EXAMPLES,
    FEWSHOT_BIAS_TEST,
    call_llm,
    mo,
    plt,
    random,
    run_bias_btn,
    set_bias_predictions,
):
    if run_bias_btn.value:
        _predictions = []
        for _trial in range(10):
            _shuffled = list(FEWSHOT_BIAS_EXAMPLES)
            random.shuffle(_shuffled)
            _p_parts = ["Classify the sentiment as Positive, Negative, or Neutral.\n"]
            for _text, _label in _shuffled:
                _p_parts.append(f'Text: "{_text}"\nSentiment: {_label}\n')
            _p_parts.append(f'Text: "{FEWSHOT_BIAS_TEST}"\nSentiment:')
            _prompt_str = "\n".join(_p_parts)
            _pred = call_llm(_prompt_str).strip().split("\n")[0]
            _predictions.append(_pred)

        set_bias_predictions(_predictions)

        _counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Other": 0}
        for _p in _predictions:
            _matched = False
            for _k in ["Positive", "Negative", "Neutral"]:
                if _k.lower() in _p.lower():
                    _counts[_k] += 1
                    _matched = True
                    break
            if not _matched:
                _counts["Other"] += 1

        _fig2, _ax2 = plt.subplots(figsize=(6, 4))
        _ax2.bar(_counts.keys(), _counts.values(), color=["#4caf50", "#f44336", "#2196f3", "#9e9e9e"])
        _ax2.set_ylabel("Count (out of 10 trials)")
        _ax2.set_title("Few-shot prediction distribution across 10 shuffled orderings")
        plt.tight_layout()

        _output = mo.vstack(
            [
                mo.as_html(_fig2),
                mo.md(f"**Predictions:** {', '.join(_predictions)}"),
                mo.md(
                    "*Reflection: How stable is the prediction across orderings? "
                    "What does this imply for deploying few-shot prompts in production?*"
                ),
            ]
        )
    else:
        _output = mo.md("*Click **Run 10 shuffled trials** to see the distribution.*")
    _output
    return


@app.cell(hide_code=True)
def _(get_bias_predictions, mo):
    # Success criterion: student must run the 10 trials and observe variance in predictions.
    # Pass = at least 2 different class labels appear across the 10 runs.
    _preds = get_bias_predictions()
    if not _preds:
        _output = mo.callout(
            mo.md("**No trials run yet.** Click **Run 10 shuffled trials** to begin."),
            kind="neutral",
        )
    else:
        _unique = set()
        for _p in _preds:
            for _k in ["Positive", "Negative", "Neutral"]:
                if _k.lower() in _p.lower():
                    _unique.add(_k)
                    break
        if len(_unique) >= 2:
            _output = mo.callout(
                mo.md(
                    f"**✅ Task 3 complete.** You observed {len(_unique)} different prediction "
                    f"classes across 10 shuffled orderings ({', '.join(sorted(_unique))}). "
                    "This variance is the key insight: a few-shot prompt is not stable across "
                    "example orderings."
                ),
                kind="success",
            )
        else:
            _label = list(_unique)[0] if _unique else "Unknown"
            _output = mo.callout(
                mo.md(
                    f"**⚠️ All 10 trials returned '{_label}'.** This model may be too confident "
                    "on this test sentence. Try re-running to get more shuffled orderings, or "
                    "consider whether the sentence is ambiguous enough to trigger variance."
                ),
                kind="warn",
            )
    _output
    return


if __name__ == "__main__":
    app.run()
