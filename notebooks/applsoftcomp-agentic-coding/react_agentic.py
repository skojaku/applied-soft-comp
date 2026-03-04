# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "litellm",
#   "langchain",
#   "langchain-community",
#   "pandas",
#   "matplotlib",
#   "duckdb",
# ]
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", title="Agentic AI: From Reasoning to Action")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import re
    import matplotlib.pyplot as plt
    return plt, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Agentic AI: From Reasoning to Action

        *The ReAct loop, LangChain tools, and context engineering through multi-agent isolation*

        /// tip | How to run this notebook
        Download this file, then open a terminal and run:

        ```
        marimo edit --sandbox react_agentic.py
        ```

        If you do not have marimo installed, install it first with `pip install marimo` or run it without installation using `uvx marimo edit --sandbox react_agentic.py`. The `--sandbox` flag creates an isolated environment and installs all dependencies automatically.
        ///

        /// note | What you'll learn in this module
        This module introduces agentic AI systems. We will explore how an agent differs from
        a chatbot by operating in a feedback loop, examine the ReAct pattern (Reason and Act)
        that structures this loop, and understand how context engineering through multi-agent
        isolation improves accuracy on complex tasks.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Configuration")
    return


@app.cell(hide_code=True)
def _(mo):
    model_input = mo.ui.text(
        value="ollama/ministral-3b:14b-cloud",
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
    config_panel = mo.vstack([
        mo.md("### LLM Configuration"),
        model_input,
        api_key_input,
        api_base_input,
        mo.md(
            "*Change any field above and all cells that use the agent will update automatically.*\n\n"
            "**Default model:** `ollama/ministral-3b:14b-cloud` — a free cloud model served through your local ollama installation. "
            "No API key is required. Run `ollama list` in your terminal to see all available models. "
            "To use a different local model, first run `ollama pull <model-name>` in your terminal, "
            "then update the model string above (e.g., `ollama/llama3.2`)."
        ),
    ])
    config_panel
    return api_base_input, api_key_input, model_input, config_panel


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
        _ollama_output = "ollama not found. Install it from https://ollama.com or use a cloud model string (e.g. openai/gpt-4o)."
        _ollama_available = False
    except Exception as _e:
        _ollama_output = f"Could not run ollama list: {_e}"
        _ollama_available = False

    if _ollama_available and _ollama_output:
        _display = mo.callout(
            mo.vstack([
                mo.md("**Available ollama models** (from `ollama list`):"),
                mo.md(f"```\n{_ollama_output}\n```"),
                mo.md("Use the model name above in the config panel as `ollama/<model-name>` (e.g. `ollama/llama3.2`)."),
            ]),
            kind="success",
        )
    elif _ollama_available:
        _display = mo.callout(
            mo.md("No local ollama models found. Run `ollama pull <model-name>` to download one, or use a cloud model string."),
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
    llm_model = model_input.value
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

    def call_llm(messages: list, stream: bool = False):
        """Call the configured LLM with a list of messages."""
        kwargs = {"model": llm_model, "messages": messages}
        if llm_api_key:
            kwargs["api_key"] = llm_api_key
        if llm_api_base:
            kwargs["api_base"] = llm_api_base
        if stream:
            kwargs["stream"] = True
        try:
            return litellm.completion(**kwargs)
        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower() or "model" in error_msg.lower():
                raise RuntimeError(
                    f"Model `{llm_model}` was not found. "
                    "Please run `ollama pull <model>` to download it, "
                    "or switch to a cloud model in the configuration panel above."
                )
            raise

    mo.accordion({
        "call_llm() helper — click to view": mo.md(
            f"*Using model: `{llm_model}`*\n\n"
            "```python\n"
            "def call_llm(messages: list, stream: bool = False):\n"
            "    # Calls the configured LLM via litellm.completion()\n"
            "    # Handles api_key, api_base, and friendly error messages\n"
            "    ...\n"
            "```"
        )
    })
    return call_llm, litellm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Section 1: Agent vs. Chatbot — A State Machine, Not a Function

        A chatbot is a function: one input, one output, done. You send a message and receive
        a reply. The interaction is stateless from the model's perspective.

        An agent is a loop. It observes the current state of the world, decides what to do,
        takes an action, and then observes the result. That new observation feeds into the
        next decision. The loop continues until the agent decides it has a final answer.

        Consider a concrete example. Ask both a chatbot and an agent: "What is the population
        of the city with the longest name in Europe?" A chatbot answers immediately from memory,
        which may be wrong or outdated. An agent pauses, decides it needs to search, calls a
        search tool, reads the result, and only then answers. The feedback loop is what makes
        verification possible.

        The figure below illustrates the difference. The chatbot is a single arrow from input
        to output. The agent is a cycle.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, plt):
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    def _make_diagram():
        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))
        _ax1.set_xlim(0, 10); _ax1.set_ylim(0, 6); _ax1.axis("off")
        _ax1.set_title("Chatbot: Input → Output", fontsize=12, fontweight="bold")
        _ax1.add_patch(plt.Rectangle((0.5, 2.5), 2.5, 1.5, facecolor="#e3f2fd", edgecolor="#1565c0", linewidth=2))
        _ax1.text(1.75, 3.25, "Input", ha="center", va="center", fontsize=11)
        _ax1.annotate("", xy=(6.5, 3.25), xytext=(3.5, 3.25), arrowprops=dict(arrowstyle="->", color="#1565c0", lw=2))
        _ax1.add_patch(plt.Rectangle((6.5, 2.5), 2.5, 1.5, facecolor="#e8f5e9", edgecolor="#2e7d32", linewidth=2))
        _ax1.text(7.75, 3.25, "Output", ha="center", va="center", fontsize=11)
        _ax2.set_xlim(0, 10); _ax2.set_ylim(0, 8); _ax2.axis("off")
        _ax2.set_title("Agent: Observe → Think → Act → Observe", fontsize=12, fontweight="bold")
        for _lbl, _x, _y, _fc, _ec in [("Observe",5,6.5,"#fff9c4","#f9a825"),("Think",8,4,"#fce4ec","#c62828"),("Act",5,1.5,"#e8f5e9","#2e7d32"),("Observe",2,4,"#e3f2fd","#1565c0")]:
            _ax2.add_patch(plt.Rectangle((_x-1.2,_y-0.6),2.4,1.2,facecolor=_fc,edgecolor=_ec,linewidth=2))
            _ax2.text(_x, _y, _lbl, ha="center", va="center", fontsize=10, fontweight="bold")
        for _x1,_y1,_x2,_y2 in [(5,5.9,8,4.6),(8,3.4,5,2.1),(5,0.9,2,3.4),(2,4.6,5,5.9)]:
            _ax2.annotate("", xy=(_x2,_y2), xytext=(_x1,_y1), arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5))
        plt.tight_layout()
        return _fig

    mo.accordion({"Chatbot vs. Agent diagram — click to view": mo.as_html(_make_diagram())})
    return FancyArrowPatch, mpatches


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Section 2: The ReAct Pattern — Reason, Then Act

        ReAct stands for Reason and Act. The idea is that the model alternates between
        reasoning about the current situation and taking a concrete action.

        The loop works as follows. The agent first reads an **Observation**: whatever the
        environment last told it. It then writes a **Thought**: a chain-of-thought reasoning
        step. From that Thought, it selects an **Action** and provides the parameters. The
        tool runs and returns a **Result**. That Result becomes the next Observation. The
        loop ends when the agent's Thought concludes that a Final Answer is ready.

        The pseudocode below shows this structure with the four labels clearly annotated.
        """
    )
    return


@app.cell(hide_code=False)
def _(mo):
    # ReAct loop: the four labels the agent alternates between each iteration.
    mo.callout(mo.md("""```
Observation: [initial question or tool result]
Thought:     [chain-of-thought reasoning step]
Action:      [tool_name]
Action Input: [tool parameters as JSON]
Observation: [tool return value]
Thought:     [reasoning based on observation]
... (repeat until done)
Thought:     I now have enough information to answer.
Final Answer: [the answer to the original question]
```
*Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023.*
"""), kind="info")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Section 3: Defining Tools — The @tool Decorator

        A tool is just a Python function with a clear docstring. The LLM never sees the
        function body. It only sees the name, the parameter types, and what the docstring says.

        Three simple tools are defined below. For each one, notice how the docstring describes
        exactly what the function does, what its parameters mean, and what it returns. Every
        word in the docstring shapes how the agent decides to call the tool.

        Try editing the docstring of one tool, then run a question that would normally use it.
        A vague or misleading docstring causes the agent to call the tool with wrong parameters
        or skip it entirely.
        """
    )
    return


@app.cell(hide_code=False)
def _():
    import datetime
    def get_current_date() -> str:
        """Return today's date as a string in YYYY-MM-DD format.
        Use this tool when the user asks what today's date is."""
        return datetime.date.today().isoformat()
    return datetime, get_current_date


@app.cell(hide_code=False)
def _():
    def evaluate_math(expression: str) -> str:
        """Evaluate a mathematical expression and return the result as a string.
        The expression must be a valid Python arithmetic expression (e.g., '2 + 2').
        Do not use this for symbolic algebra — only numeric calculations."""
        try:
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression): return "Error: disallowed characters."
            return str(eval(expression, {"__builtins__": {}}))  # noqa: S307
        except Exception as e: return f"Error: {e}"
    return (evaluate_math,)


@app.cell(hide_code=False)
def _():
    def define_word(word: str) -> str:
        """Look up the definition of a common English word. Returns one sentence.
        Use this when the user asks what a word means. Cannot look up jargon."""
        _d = {"serendipity": "The occurrence of fortunate events by chance.",
              "ephemeral": "Lasting for a very short time; transitory.",
              "algorithm": "A step-by-step procedure for solving a problem.",
              "entropy": "A measure of disorder or randomness in a system.",
              "heuristic": "A practical approach that is good enough for the goal."}
        return _d.get(word.lower(), f"Definition not found for '{word}'.")
    return (define_word,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "```python\n"
        "def get_current_date() -> str:\n"
        '    """Return today\'s date as a string in YYYY-MM-DD format.\n'
        "    Use this tool when the user asks what today's date is.\"\"\"\n"
        "    ...\n\n"
        "def evaluate_math(expression: str) -> str:\n"
        '    """Evaluate a mathematical expression and return the result.\n'
        "    The expression must be a valid Python arithmetic expression.\"\"\"\n"
        "    ...\n\n"
        "def define_word(word: str) -> str:\n"
        '    """Look up the definition of a common English word.\n'
        "    Returns a brief one-sentence definition.\"\"\"\n"
        "    ...\n"
        "```"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Section 4: A ReAct Agent in Action — Live Trace

        Let us watch the agent work. Type any question in the box below. The three tools
        from the previous section are available. As the agent reasons through the problem,
        each step of the ReAct trace appears on screen: Thought, Action, Action Input, and
        Observation on separate lines.

        Try asking a question that requires at least two tool calls, for example: "What is
        today's date and how many days until January 1st of next year?" Then re-read the
        trace and identify exactly where the agent changed direction based on what a tool
        returned.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    agent_question = mo.ui.text_area(
        value="What is today's date, and what is the definition of serendipity?",
        label="Ask the agent a question",
        full_width=True,
        rows=3,
    )
    run_agent_btn = mo.ui.run_button(label="Run agent")
    mo.vstack([agent_question, run_agent_btn])
    return agent_question, run_agent_btn


@app.cell(hide_code=True)
def _(define_word, evaluate_math, get_current_date):
    import json
    TOOLS = {"get_current_date": get_current_date, "evaluate_math": evaluate_math, "define_word": define_word}
    TOOL_DESC = "Tools: get_current_date(), evaluate_math(expr), define_word(word).\nFormat: Thought/Action/Action Input/Final Answer"
    return TOOL_DESC, TOOLS, json


@app.cell(hide_code=True)
def _(TOOL_DESC, TOOLS, call_llm, re):
    def run_react_agent(question, max_steps=8):
        # ReAct loop: Thought → Action → Observation → repeat until Final Answer
        trace, msgs = [], [{"role": "system", "content": TOOL_DESC}, {"role": "user", "content": question}]
        for _ in range(max_steps):
            try: text = call_llm(msgs).choices[0].message.content
            except Exception as e: trace.append({"type": "error", "content": str(e)}); break
            msgs.append({"role": "assistant", "content": text})
            if "Final Answer:" in text: trace.append({"type": "final", "content": text[text.index("Final Answer:")+13:].strip()}); break
            am, im = re.search(r"Action:\s*(.+)", text), re.search(r"Action Input:\s*(.+)", text)
            if tm := re.search(r"Thought:\s*(.+?)(?:\nAction|$)", text, re.DOTALL): trace.append({"type": "thought", "content": tm.group(1).strip()})
            if am and im:
                tn, ti = am.group(1).strip(), im.group(1).strip()
                try: obs = TOOLS[tn](ti) if tn in TOOLS else f"Tool '{tn}' not found."
                except Exception as e: obs = f"Error: {e}"
                trace += [{"type": "action", "tool": tn, "input": ti}, {"type": "observation", "content": obs}]
                msgs.append({"role": "user", "content": f"Observation: {obs}"})
            else: trace.append({"type": "thought", "content": text}); break
        return trace
    return (run_react_agent,)


@app.cell(hide_code=True)
def _(agent_question, mo, run_agent_btn, run_react_agent):
    if run_agent_btn.value:
        _trace = run_react_agent(agent_question.value)
        _parts = [mo.md(f"### ReAct Trace ({len(_trace)} steps)")]
        for _s in _trace:
            if _s["type"] == "thought":
                _parts.append(mo.callout(mo.md(f"**Thought:** {_s['content']}"), kind="info"))
            elif _s["type"] == "action":
                _parts.append(mo.callout(mo.md(f"**Action:** `{_s['tool']}`\n\n**Input:** `{_s['input']}`"), kind="warn"))
            elif _s["type"] == "observation":
                _parts.append(mo.callout(mo.md(f"**Observation:** {_s['content']}"), kind="neutral"))
            elif _s["type"] == "final":
                _parts.append(mo.callout(mo.md(f"**Final Answer:** {_s['content']}"), kind="success"))
            elif _s["type"] == "error":
                _parts.append(mo.callout(mo.md(f"**Error:** {_s['content']}"), kind="danger"))
        _output = mo.vstack(_parts)
    else:
        _output = mo.md("*Click **Run agent** to watch the ReAct loop unfold.*")
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Section 5: Data Detective — Interrogating the Titanic Dataset

        Let us give the agent a harder job. The Titanic dataset is loaded below, and the agent
        has four tools available: one that returns column names and types, one that returns sample
        rows, one that runs a SQL query via DuckDB, and one that returns summary statistics for
        a column.

        Before running the agent, read each tool's docstring. The system prompt instructs the
        agent to always verify its answer with at least one tool call. Pick a question from the
        buttons below, or type your own. After the agent answers, read the trace and check whether
        its reasoning actually matches the tool outputs it received.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    titanic_question = mo.ui.text_area(
        value="What was the survival rate for each passenger class?",
        label="Ask the Titanic data detective",
        full_width=True,
        rows=3,
    )

    preset_q1 = mo.ui.run_button(label="What was the overall survival rate?")
    preset_q2 = mo.ui.run_button(label="What is the average age of survivors vs. non-survivors?")
    preset_q3 = mo.ui.run_button(label="Which embarkation port had the highest survival rate?")

    run_titanic_btn = mo.ui.run_button(label="Run data detective")

    mo.vstack([
        mo.md("**Quick questions:**"),
        mo.hstack([preset_q1, preset_q2, preset_q3]),
        titanic_question,
        run_titanic_btn,
    ])
    return (
        preset_q1,
        preset_q2,
        preset_q3,
        run_titanic_btn,
        titanic_question,
    )


@app.cell(hide_code=True)
def _(mo):
    import pandas as pd
    import duckdb

    try:
        titanic_df = pd.read_csv(
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        )
    except Exception:
        import io
        _csv = (
            "PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked\n"
            "1,0,3,Braund Mr. Owen Harris,male,22,1,0,A/5 21171,7.25,,S\n"
            "2,1,1,Cumings Mrs. John Bradley,female,38,1,0,PC 17599,71.2833,C85,C\n"
            "3,1,3,Heikkinen Miss. Laina,female,26,0,0,STON/O2. 3101282,7.925,,S\n"
            "4,1,1,Futrelle Mrs. Jacques Heath,female,35,1,0,113803,53.1,C123,S\n"
            "5,0,3,Allen Mr. William Henry,male,35,0,0,373450,8.05,,S\n"
        )
        titanic_df = pd.read_csv(io.StringIO(_csv))

    mo.accordion({
        f"Dataset loaded: {len(titanic_df)} rows × {len(titanic_df.columns)} columns — click to preview": mo.md(
            f"```\n{titanic_df.head(3).to_string()}\n```"
        ),
    })
    return duckdb, pd, titanic_df


@app.cell(hide_code=True)
def _(duckdb, mo, titanic_df):
    def inspect_schema() -> str:
        """Return column names and data types. Use this first before any query."""
        return titanic_df.dtypes.to_string()

    def get_sample_rows(n: str = "5") -> str:
        """Return the first n rows as a string. n must be a string integer (e.g., '5')."""
        try: return titanic_df.head(int(n)).to_string()
        except Exception as e: return f"Error: {e}"

    def run_sql_query(query: str) -> str:
        """Run SQL against the Titanic dataset (table: titanic_df). Standard SQL only.
        Example: SELECT AVG(Age) FROM titanic_df WHERE Survived = 1
        Always check inspect_schema first to confirm column names."""
        try: return duckdb.query(query).df().to_string(index=False)
        except Exception as e: return f"SQL Error: {e}. Available columns: {list(titanic_df.columns)}"

    def get_summary_stats(column: str) -> str:
        """Return descriptive statistics for a numeric column (case-sensitive column name).
        Use inspect_schema first to confirm the exact column name."""
        if column not in titanic_df.columns:
            return f"Column '{column}' not found. Available: {list(titanic_df.columns)}"
        return titanic_df[column].describe().to_string()

    TITANIC_TOOLS = {"inspect_schema": lambda _: inspect_schema(), "get_sample_rows": get_sample_rows,
                     "run_sql_query": run_sql_query, "get_summary_stats": get_summary_stats}
    TITANIC_SYSTEM_PROMPT = (
        "You are a data detective analyzing the Titanic dataset. "
        "You MUST verify every claim with at least one tool call before giving your Final Answer. "
        "Never guess — always query the data.\n\n"
        "Available tools: inspect_schema(), get_sample_rows(n), run_sql_query(query), get_summary_stats(column)\n\n"
        "Format:\nThought: [reasoning]\nAction: [tool]\nAction Input: [input]\n"
        "Observation: [system fills this]\n...\nFinal Answer: [answer]"
    )
    mo.accordion({
        "Titanic tool definitions — click to view": mo.md(
            "*Four tools ready: `inspect_schema`, `get_sample_rows`, `run_sql_query`, `get_summary_stats`.*\n\n"
            "```python\n"
            "def inspect_schema() -> str:\n"
            '    """Return column names and data types. Use this first."""\n'
            "    ...\n\n"
            "def run_sql_query(query: str) -> str:\n"
            '    """Run SQL against titanic_df (DuckDB). Check schema first."""\n'
            "    ...\n"
            "```"
        )
    })
    return TITANIC_SYSTEM_PROMPT, TITANIC_TOOLS, get_sample_rows, get_summary_stats, inspect_schema, run_sql_query


@app.cell(hide_code=True)
def _(TITANIC_SYSTEM_PROMPT, TITANIC_TOOLS, call_llm, mo, re):
    def run_titanic_agent(question: str, max_steps: int = 10) -> list:
        trace = []
        messages = [{"role": "system", "content": TITANIC_SYSTEM_PROMPT}, {"role": "user", "content": question}]
        for _ in range(max_steps):
            try: text = call_llm(messages).choices[0].message.content
            except Exception as e: trace.append({"type": "error", "content": str(e)}); break
            messages.append({"role": "assistant", "content": text})
            if "Final Answer:" in text:
                trace.append({"type": "final", "content": text[text.index("Final Answer:")+13:].strip()}); break
            am = re.search(r"Action:\s*(.+)", text); im = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
            tm = re.search(r"Thought:\s*(.+?)(?:\nAction|$)", text, re.DOTALL)
            if tm: trace.append({"type": "thought", "content": tm.group(1).strip()})
            if am and im:
                tool_name = am.group(1).strip(); tool_input = im.group(1).strip().split("\n")[0].strip()
                trace.append({"type": "action", "tool": tool_name, "input": tool_input})
                try: obs = TITANIC_TOOLS[tool_name](tool_input) if tool_name in TITANIC_TOOLS else f"Tool '{tool_name}' not found."
                except Exception as e: obs = f"Error: {e}"
                trace.append({"type": "observation", "content": obs})
                messages.append({"role": "user", "content": f"Observation: {obs}"})
            else: trace.append({"type": "thought", "content": text}); break
        return trace

    mo.accordion({
        "run_titanic_agent() implementation — click to view": mo.md(
            "*Implements the same Thought → Action → Observation loop as the ReAct demo.*\n\n"
            "```python\n"
            "def run_titanic_agent(question: str, max_steps: int = 10) -> list:\n"
            "    # Builds messages, calls LLM, parses Thought/Action/Observation\n"
            "    # Returns a trace list with typed dicts\n"
            "    ...\n"
            "```"
        )
    })
    return run_titanic_agent,


@app.cell(hide_code=True)
def _(mo, preset_q1, preset_q2, preset_q3, run_titanic_agent, run_titanic_btn, titanic_question):
    _q = None
    if preset_q1.value: _q = "What was the overall survival rate?"
    elif preset_q2.value: _q = "What is the average age of survivors vs. non-survivors?"
    elif preset_q3.value: _q = "Which embarkation port had the highest survival rate?"
    elif run_titanic_btn.value: _q = titanic_question.value

    if _q:
        _trace = run_titanic_agent(_q)
        _parts = [mo.md(f"### Titanic Agent Trace — *{_q}*")]
        for _item in _trace:
            if _item["type"] == "thought":
                _parts.append(mo.callout(mo.md(f"**Thought:** {_item['content']}"), kind="info"))
            elif _item["type"] == "action":
                _parts.append(mo.callout(mo.md(f"**Action:** `{_item['tool']}`\n\n**Input:** `{_item['input']}`"), kind="warn"))
            elif _item["type"] == "observation":
                _parts.append(mo.callout(mo.md(f"**Observation:**\n```\n{_item['content']}\n```"), kind="neutral"))
            elif _item["type"] == "final":
                _parts.append(mo.callout(mo.md(f"**Final Answer:** {_item['content']}"), kind="success"))
            elif _item["type"] == "error":
                _parts.append(mo.callout(mo.md(f"**Error:** {_item['content']}"), kind="danger"))
        _output = mo.vstack(_parts)
    else:
        _output = mo.md("*Select a preset question or type your own and click **Run data detective**.*")
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Section 6: Context Engineering — Multi-Agent Isolation for Literature Verification

        When you give a single agent a long list of things to verify, the references start to
        bleed into each other. The agent may use a detail from one paper to justify a claim
        about another. This is context confusion, and it grows worse as the context window fills.

        Below are five academic references: three real, two fabricated. The real papers are
        well-known landmarks in machine learning. The fabricated papers use plausible-sounding
        author names, venues, and titles. A real LLM can be fooled by the surface plausibility.

        First, run the monolithic agent. It receives all five references in a single prompt and
        is asked to assess each one. Watch whether it makes confident but incorrect statements,
        and pay attention to where its verdicts bleed across references. The token counter shows
        how much of the context window this single call consumes.

        Then run the five isolated sub-agents. Each sub-agent receives only one reference. The
        token count per sub-agent is much lower, and the verdicts are more reliable. The key
        insight is not just accuracy but the reason for accuracy: a clean context window gives
        the model no opportunity to confuse one paper with another.

        *Reflection: Did the monolithic agent make any wrong verdicts? Did it mix up authors
        or venues? Which approach was more accurate, and why?*
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    REFERENCES = [
        {
            "id": "ref1",
            "citation": "LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.",
            "real": True,
            "verdict": "Real — landmark survey paper with DOI 10.1038/nature14539.",
        },
        {
            "id": "ref2",
            "citation": "Vaswani, A., et al. (2017). Attention is all you need. NeurIPS, 30.",
            "real": True,
            "verdict": "Real — the original Transformer paper.",
        },
        {
            "id": "ref3",
            "citation": "Yao, S., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023.",
            "real": True,
            "verdict": "Real — the ReAct paper we referenced earlier.",
        },
        {
            "id": "ref4",
            "citation": "Chen, M., & Park, J. (2021). Recursive attention sparsification for efficient transformer inference. ICML Workshop on Efficient Deep Learning, pp. 112-128.",
            "real": False,
            "verdict": "Fabricated — no such paper exists in ICML 2021 workshops.",
        },
        {
            "id": "ref5",
            "citation": "Weston, J., Bordes, A., & Chopra, S. (2023). Memory-augmented neural networks for continual knowledge integration. Journal of Machine Learning Research, 24(1), 1-42.",
            "real": False,
            "verdict": "Fabricated — these authors exist but this paper does not.",
        },
    ]

    run_monolithic_btn = mo.ui.run_button(label="Run monolithic agent (all 5 references)")
    run_multiagent_btn = mo.ui.run_button(label="Run 5 isolated sub-agents (parallel)")

    mo.vstack([
        mo.md("### References to verify"),
        mo.md("\n\n".join(f"**[{r['id']}]** {r['citation']}" for r in REFERENCES)),
        mo.hstack([run_monolithic_btn, run_multiagent_btn]),
    ])
    return REFERENCES, run_monolithic_btn, run_multiagent_btn


@app.cell(hide_code=True)
def _(REFERENCES, call_llm, mo, re, run_monolithic_btn, run_multiagent_btn):
    from concurrent.futures import ThreadPoolExecutor

    VERIFY_SYSTEM = (
        "You are a fact-checker for academic references. "
        "For each reference, state whether it is REAL or FABRICATED and give a brief reason. "
        "Be concise. Label each verdict clearly with the reference ID, e.g.: "
        "[ref1] REAL — ... [ref2] FABRICATED — ..."
    )

    def _parse_verdicts(text: str, refs: list) -> dict:
        """Extract per-reference verdicts from a monolithic response."""
        verdicts = {}
        for ref in refs:
            rid = ref["id"]
            # Look for patterns like [ref1] REAL or ref1: FABRICATED
            pattern = rf"\[?{rid}\]?\s*[:\-]?\s*(REAL|FABRICATED|real|fabricated|Real|Fabricated)"
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                label = m.group(1).upper()
                verdicts[rid] = label == "REAL"
            else:
                verdicts[rid] = None  # could not parse
        return verdicts

    def verify_single(ref: dict) -> dict:
        """Verify one reference in isolation — a separate sub-agent with its own clean context."""
        messages = [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": f"Verify this reference: {ref['citation']}"},
        ]
        try:
            resp = call_llm(messages)
            content = resp.choices[0].message.content
            tokens_used = getattr(resp.usage, "total_tokens", None)
            return {
                "id": ref["id"],
                "verdict": content,
                "real": ref["real"],
                "tokens": tokens_used,
            }
        except Exception as e:
            return {"id": ref["id"], "verdict": f"Error: {e}", "real": ref["real"], "tokens": None}

    if run_monolithic_btn.value:
        _all_refs = "\n".join(f"[{r['id']}] {r['citation']}" for r in REFERENCES)
        _messages = [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": (
                "Verify each of the following five references. "
                "For each one, state REAL or FABRICATED and give a brief reason. "
                "Label each verdict with its reference ID.\n\n" + _all_refs
            )},
        ]
        try:
            _resp = call_llm(_messages)
            _mono_text = _resp.choices[0].message.content
            _mono_tokens = getattr(_resp.usage, "total_tokens", "unknown")

            # Parse verdicts and compare to ground truth
            _parsed = _parse_verdicts(_mono_text, REFERENCES)
            _verdict_rows = []
            _correct_count = 0
            for _ref in REFERENCES:
                _rid = _ref["id"]
                _gt_real = _ref["real"]
                _predicted_real = _parsed.get(_rid)
                if _predicted_real is None:
                    _icon = "❓"
                    _note = "Could not parse verdict"
                elif _predicted_real == _gt_real:
                    _icon = "✅"
                    _correct_count += 1
                    _note = "Correct"
                else:
                    _icon = "❌"
                    _note = "Wrong — context confusion?"
                _gt_label = "REAL" if _gt_real else "FABRICATED"
                _verdict_rows.append(f"{_icon} **{_rid}** (ground truth: {_gt_label}) — {_note}")

            _accuracy = f"{_correct_count}/{len(REFERENCES)}"
            _parts = [
                mo.md("### Monolithic agent — all five references in one prompt"),
                mo.md(f"*Token count for this call: **{_mono_tokens}***"),
                mo.callout(mo.md(_mono_text), kind="info"),
                mo.md("**Verdict check against ground truth:**"),
                mo.callout(
                    mo.md(
                        "\n\n".join(_verdict_rows)
                        + f"\n\n**Accuracy: {_accuracy}**"
                    ),
                    kind="neutral",
                ),
                mo.md("**Ground truth:**"),
                mo.md("\n\n".join(f"**{r['id']}:** {r['verdict']}" for r in REFERENCES)),
            ]
            _output = mo.vstack(_parts)
        except Exception as _e:
            _output = mo.callout(mo.md(f"**Error:** {_e}"), kind="danger")

    elif run_multiagent_btn.value:
        try:
            # Run five isolated sub-agents in parallel — each gets its own clean context window
            with ThreadPoolExecutor(max_workers=5) as executor:
                _futures = [executor.submit(verify_single, r) for r in REFERENCES]
                _results = [f.result() for f in _futures]
            _total_tokens = sum(r["tokens"] for r in _results if r["tokens"] is not None)
            _token_note = f"Total tokens across all sub-agents: **{_total_tokens}**" if _total_tokens else ""

            _parts = [
                mo.md("### Multi-agent results — one isolated sub-agent per reference"),
                mo.md(f"*{_token_note}*") if _token_note else mo.md(""),
            ]
            _correct_count = 0
            for _r in _results:
                _gt = next(ref for ref in REFERENCES if ref["id"] == _r["id"])
                # Parse the single-reference verdict
                _is_real_predicted = "REAL" in _r["verdict"].upper() and "FABRICATED" not in _r["verdict"].upper()
                _correct = _is_real_predicted == _gt["real"]
                if _correct:
                    _correct_count += 1
                _icon = "✅" if _correct else "❌"
                _gt_label = "Real" if _gt["real"] else "Fabricated"
                _tok = f"({_r['tokens']} tokens)" if _r.get("tokens") else ""
                _parts.append(mo.vstack([
                    mo.md(f"**{_r['id']}** — Ground truth: {_gt_label} {_icon} {_tok}"),
                    mo.callout(mo.md(_r["verdict"]), kind="info"),
                ]))
            _accuracy = f"{_correct_count}/{len(REFERENCES)}"
            _parts.append(mo.callout(
                mo.md(f"**Multi-agent accuracy: {_accuracy}** (compare to monolithic above)"),
                kind="neutral",
            ))
            _parts.append(mo.md("**Ground truth:**"))
            _parts.append(mo.md("\n\n".join(f"**{r['id']}:** {r['verdict']}" for r in REFERENCES)))
            _output = mo.vstack(_parts)
        except Exception as _e:
            _output = mo.callout(mo.md(f"**Error:** {_e}"), kind="danger")
    else:
        _output = mo.md("*Click a button above to run the monolithic or multi-agent demo.*")
    _output
    return VERIFY_SYSTEM, ThreadPoolExecutor, _parse_verdicts, verify_single


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("## Student Task: Extend the Data Detective"),
        mo.callout(
            mo.md(
                r"""
**Try it yourself**

Add two new tools to the Titanic agent. The first returns a cross-tabulation (pivot table)
of two columns. The second filters rows by a column value and returns a count.

After adding both tools, pose this question to the agent: "Among female passengers over
30 years old, which ticket class had the highest survival rate, and how does it compare
to males in the same age group?"

The agent must chain at least three tool calls to answer correctly. A ground-truth cell
verifies the numerical answer.

**Extension:** Modify the system prompt to instruct the agent to always question its
first answer and run one additional verification tool call. Does this improve accuracy?
                """
            ),
            kind="info",
        ),
        mo.callout(
            mo.md(
                "**Hint:** Write the docstring first. Describe exactly what the function "
                "returns, what the parameters mean, and what format the output is in. "
                "The LLM will call your tool exactly as described — be precise."
            ),
            kind="neutral",
        ),
        mo.accordion({
            "Show more (detailed hint)": mo.md(
                "For `cross_tabulation`, the input should be two column names. Specify in "
                "the docstring that the input is a comma-separated string (e.g., 'Sex,Survived') "
                "and that the output is a formatted table. For `filter_and_count`, the input "
                "should be a pandas query string (e.g., 'Sex == \"female\" and Age > 30'). "
                "Mention in the docstring that the user should check `inspect_schema` first "
                "to confirm available column names and their exact spelling. A tool with a "
                "vague docstring will be called incorrectly or skipped entirely."
            ),
        }),
    ])
    return


@app.cell(hide_code=False)
def _(pd, titanic_df):
    def cross_tabulation(col1_col2: str) -> str:
        """Return a cross-tabulation of two columns as a string.
        Input: two column names separated by a comma, e.g., 'Sex,Survived'.
        Use this to compare counts across two categorical variables."""
        try:
            cols = [c.strip() for c in col1_col2.split(",")]
            return pd.crosstab(titanic_df[cols[0]], titanic_df[cols[1]]).to_string()
        except Exception as e:
            return f"Error: {e}. Available columns: {list(titanic_df.columns)}"

    def filter_and_count(condition: str) -> str:
        """Filter the Titanic dataset and return the count of matching rows.
        Input: a pandas query string, e.g., 'Sex == "female" and Age > 30'.
        Always check inspect_schema first to confirm column names."""
        try:
            return f"{len(titanic_df.query(condition))} rows match: {condition}"
        except Exception as e:
            return f"Error: {e}. Available columns: {list(titanic_df.columns)}"

    return cross_tabulation, filter_and_count


@app.cell(hide_code=True)
def _(mo):
    run_extended_btn = mo.ui.run_button(label="Run extended agent with target question")
    mo.vstack([
        mo.callout(
            mo.md(
                "**Target question:** Among female passengers over 30 years old, which ticket "
                "class had the highest survival rate, and how does it compare to males in the same age group?"
            ),
            kind="info",
        ),
        run_extended_btn,
    ])
    return (run_extended_btn,)


@app.cell(hide_code=True)
def _(TITANIC_SYSTEM_PROMPT, TITANIC_TOOLS, call_llm, cross_tabulation, filter_and_count, mo, re, titanic_df):
    EXTENDED_TOOLS = dict(TITANIC_TOOLS)
    EXTENDED_TOOLS["cross_tabulation"] = cross_tabulation
    EXTENDED_TOOLS["filter_and_count"] = filter_and_count
    TARGET_QUESTION = (
        "Among female passengers over 30 years old, which ticket class had the highest survival rate, "
        "and how does it compare to males in the same age group?"
    )

    def run_extended_agent(question: str, max_steps: int = 12) -> list:
        trace = []
        ext_system = TITANIC_SYSTEM_PROMPT + "\n5. cross_tabulation(col1,col2) — Cross-tabulate two columns.\n6. filter_and_count(condition) — Count rows matching a pandas query condition."
        messages = [{"role": "system", "content": ext_system}, {"role": "user", "content": question}]
        for _ in range(max_steps):
            try: text = call_llm(messages).choices[0].message.content
            except Exception as e: trace.append({"type": "error", "content": str(e)}); break
            messages.append({"role": "assistant", "content": text})
            if "Final Answer:" in text:
                trace.append({"type": "final", "content": text[text.index("Final Answer:")+13:].strip()}); break
            am = re.search(r"Action:\s*(.+)", text); im = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
            tm = re.search(r"Thought:\s*(.+?)(?:\nAction|$)", text, re.DOTALL)
            if tm: trace.append({"type": "thought", "content": tm.group(1).strip()})
            if am and im:
                tn = am.group(1).strip(); ti = im.group(1).strip().split("\n")[0].strip()
                trace.append({"type": "action", "tool": tn, "input": ti})
                try: obs = EXTENDED_TOOLS[tn](ti) if tn in EXTENDED_TOOLS else f"Tool '{tn}' not found."
                except Exception as e: obs = f"Error: {e}"
                trace.append({"type": "observation", "content": obs})
                messages.append({"role": "user", "content": f"Observation: {obs}"})
            else: trace.append({"type": "thought", "content": text}); break
        return trace

    # Ground truth
    _f30 = titanic_df[(titanic_df["Sex"] == "female") & (titanic_df["Age"] > 30)]
    _m30 = titanic_df[(titanic_df["Sex"] == "male") & (titanic_df["Age"] > 30)]
    best_f_class = int(_f30.groupby("Pclass")["Survived"].mean().idxmax())
    best_f_rate = float(_f30.groupby("Pclass")["Survived"].mean().max())
    best_m_class = int(_m30.groupby("Pclass")["Survived"].mean().idxmax())
    best_m_rate = float(_m30.groupby("Pclass")["Survived"].mean().max())

    mo.accordion({
        "run_extended_agent() implementation & ground truth — click to view": mo.md(
            "*Agent uses all 6 tools including cross_tabulation and filter_and_count.*\n\n"
            f"Ground truth: Female → Class {best_f_class} ({best_f_rate:.1%}), Male → Class {best_m_class} ({best_m_rate:.1%})"
        )
    })
    return EXTENDED_TOOLS, TARGET_QUESTION, best_f_class, best_f_rate, best_m_class, best_m_rate, run_extended_agent


@app.cell(hide_code=True)
def _(TARGET_QUESTION, best_f_class, best_f_rate, best_m_class, best_m_rate, mo, run_extended_agent, run_extended_btn):
    if run_extended_btn.value:
        _etrace = run_extended_agent(TARGET_QUESTION)
        _eparts = [mo.md("### Extended Agent Trace")]
        _final_answer = ""
        for _item in _etrace:
            if _item["type"] == "thought":
                _eparts.append(mo.callout(mo.md(f"**Thought:** {_item['content']}"), kind="info"))
            elif _item["type"] == "action":
                _eparts.append(mo.callout(mo.md(f"**Action:** `{_item['tool']}`\n\n**Input:** `{_item['input']}`"), kind="warn"))
            elif _item["type"] == "observation":
                _eparts.append(mo.callout(mo.md(f"**Observation:**\n```\n{_item['content']}\n```"), kind="neutral"))
            elif _item["type"] == "final":
                _final_answer = _item["content"]
                _eparts.append(mo.callout(mo.md(f"**Final Answer:** {_item['content']}"), kind="success"))
            elif _item["type"] == "error":
                _eparts.append(mo.callout(mo.md(f"**Error:** {_item['content']}"), kind="danger"))
        _gt = f"Ground truth: Female → **Class {best_f_class}** ({best_f_rate:.1%}), Male → **Class {best_m_class}** ({best_m_rate:.1%})"
        _eparts.append(mo.callout(mo.md(_gt), kind="neutral"))
        if _final_answer:
            _fl = _final_answer.lower()
            _fi = "✅" if any(s in _fl for s in [f"class {best_f_class}", f" {best_f_class} "]) else "❌"
            _mi = "✅" if any(s in _fl for s in [f"class {best_m_class}", f" {best_m_class} "]) else "❌"
            _pass = _fi == "✅" and _mi == "✅"
            _eparts.append(mo.callout(mo.md(f"{_fi} Female Class {best_f_class}  {_mi} Male Class {best_m_class}\n\n**{'PASS' if _pass else 'FAIL'}**"), kind="success" if _pass else "danger"))
        _output = mo.vstack(_eparts)
    else:
        _output = mo.md("*Click **Run extended agent with target question** to run the agent and see the ground-truth verification.*")
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("## Thought Experiment: When Does the Loop Break?"),
        mo.callout(
            mo.md(
                r"""
**Try it yourself**

A broken query tool is defined below that always raises a ValueError with a vague
error message. Run the agent on a simple question and watch what happens. Does the
agent retry? Does it give up? Does it hallucinate an answer after failing?

Then edit the error message to be more informative: name the available columns and
explain what went wrong. Run the agent again and compare.
                """
            ),
            kind="info",
        ),
        mo.callout(
            mo.md(
                "**Hint:** Agents learn from their observations. A tool that says \"Error\" "
                "gives the agent nothing to work with. A more informative message lets the "
                "agent self-correct and try a different approach."
            ),
            kind="neutral",
        ),
        mo.accordion({
            "Show more (detailed hint)": mo.md(
                "Change the error message from `raise ValueError(\"Error\")` to something like: "
                "`raise ValueError(\"column 'Age' not found. Available columns: Survived, Pclass, "
                "Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked.\")`. "
                "Then re-run the agent and watch whether it uses that information to try a "
                "corrected query. The reflection question is: what does this tell you about "
                "the relationship between tool design and agent reliability?"
            ),
        }),
    ])
    return


@app.cell(hide_code=False)
def _(TITANIC_TOOLS):
    # ✏️ Edit the error message below — the agent learns from what your tool tells it.
    def broken_query_tool(query: str) -> str:
        """Run a SQL query against the Titanic dataset.
        Edit the error message below to make it more informative."""
        raise ValueError("Error")  # TODO: make this error message more helpful

    BROKEN_TOOLS = dict(TITANIC_TOOLS)
    BROKEN_TOOLS["run_sql_query"] = broken_query_tool
    BROKEN_QUESTION = "What is the average fare paid by first-class passengers?"
    return BROKEN_QUESTION, BROKEN_TOOLS, broken_query_tool


@app.cell(hide_code=True)
def _(mo):
    run_broken_btn = mo.ui.run_button(label="Run agent with broken tool")
    mo.vstack([
        mo.callout(mo.md("**Question:** What is the average fare paid by first-class passengers?"), kind="info"),
        run_broken_btn,
    ])
    return (run_broken_btn,)


@app.cell(hide_code=True)
def _(BROKEN_TOOLS, call_llm, mo, re):
    def run_broken_agent(question: str, max_steps: int = 6) -> list:
        trace = []
        sys_prompt = ("You are a data analyst. Use run_sql_query(query) to answer about titanic_df.\n"
                      "Format:\nThought: ...\nAction: run_sql_query\nAction Input: [SQL]\nFinal Answer: ...")
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": question}]
        for _ in range(max_steps):
            try: text = call_llm(messages).choices[0].message.content
            except Exception as e: trace.append({"type": "error", "content": str(e)}); break
            messages.append({"role": "assistant", "content": text})
            if "Final Answer:" in text:
                trace.append({"type": "final", "content": text[text.index("Final Answer:")+13:].strip()}); break
            am, im = re.search(r"Action:\s*(.+)", text), re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
            if tm := re.search(r"Thought:\s*(.+?)(?:\nAction|$)", text, re.DOTALL):
                trace.append({"type": "thought", "content": tm.group(1).strip()})
            if am and im:
                tn, ti = am.group(1).strip(), im.group(1).strip().split("\n")[0]
                trace.append({"type": "action", "tool": tn, "input": ti})
                fn = BROKEN_TOOLS.get(tn)
                try: obs = fn(ti) if fn else f"Tool '{tn}' not found."
                except Exception as e: obs = str(e)
                trace.append({"type": "observation", "content": obs})
                messages.append({"role": "user", "content": f"Observation: {obs}"})
            else: trace.append({"type": "thought", "content": text}); break
        return trace

    mo.accordion({
        "run_broken_agent() implementation — click to view": mo.md(
            "*Runs the same ReAct loop but with the broken tool in place of the working SQL tool.*"
        )
    })
    return (run_broken_agent,)


@app.cell(hide_code=True)
def _(mo, run_broken_agent, run_broken_btn):
    _BQ = "What is the average fare paid by first-class passengers?"
    if run_broken_btn.value:
        _btrace = run_broken_agent(_BQ)
        _bparts = [mo.md("### Broken-tool agent trace")]
        for _item in _btrace:
            if _item["type"] == "thought":
                _bparts.append(mo.callout(mo.md(f"**Thought:** {_item['content']}"), kind="info"))
            elif _item["type"] == "action":
                _bparts.append(mo.callout(
                    mo.md(f"**Action:** `{_item['tool']}`\n\n**Input:** `{_item['input']}`"),
                    kind="warn",
                ))
            elif _item["type"] == "observation":
                _bparts.append(mo.callout(mo.md(f"**Observation:** {_item['content']}"), kind="danger"))
            elif _item["type"] == "final":
                _bparts.append(mo.callout(mo.md(f"**Final Answer:** {_item['content']}"), kind="success"))
            elif _item["type"] == "error":
                _bparts.append(mo.callout(mo.md(f"**Error:** {_item['content']}"), kind="danger"))

        _bparts.append(mo.md(
            "*Reflection: Did the agent retry? Did it hallucinate an answer? "
            "Now edit the error message in `broken_query_tool` above to be more informative and run again.*"
        ))
        _output = mo.vstack(_bparts)
    else:
        _output = mo.md("*Click **Run agent with broken tool** to watch what happens.*")
    _output
    return


@app.cell(hide_code=True)
def _(broken_query_tool, mo):
    # Success criterion: the student must edit broken_query_tool so its error message
    # is more informative. Pass = the raised ValueError message is longer than 10
    # characters AND mentions at least one Titanic column name.
    _TITANIC_COLS = {"survived", "pclass", "name", "sex", "age", "sibsp", "parch",
                     "ticket", "fare", "cabin", "embarked"}
    try:
        broken_query_tool("SELECT 1")
        _err_msg = ""
    except ValueError as _e:
        _err_msg = str(_e)
    except Exception as _e:
        _err_msg = str(_e)

    _is_default = _err_msg.strip().lower() == "error"
    _is_long = len(_err_msg.strip()) > 10
    _mentions_col = any(col in _err_msg.lower() for col in _TITANIC_COLS)

    if _is_default:
        _output = mo.callout(
            mo.md(
                "**❌ Error message not yet improved.** The tool still raises `ValueError('Error')`. "
                "Edit the `raise ValueError(...)` line above to include a helpful description: "
                "name the available columns and explain what went wrong."
            ),
            kind="danger",
        )
    elif _is_long and _mentions_col:
        _output = mo.callout(
            mo.md(
                "**✅ Task 5 complete.** Your error message is informative: it is long enough "
                "and mentions at least one Titanic column name. Run the agent again to see "
                "whether it can now self-correct using your improved error message."
            ),
            kind="success",
        )
    elif _is_long:
        _output = mo.callout(
            mo.md(
                "**⚠️ Better, but not quite.** Your error message is longer than the default, "
                "but it does not name any Titanic column (Survived, Pclass, Name, Sex, Age, "
                "SibSp, Parch, Ticket, Fare, Cabin, Embarked). Add the column list so the "
                "agent knows what to query."
            ),
            kind="warn",
        )
    else:
        _output = mo.callout(
            mo.md(
                "**⚠️ Error message too short.** Try to write a message that explains what "
                "went wrong and lists the available columns."
            ),
            kind="warn",
        )
    _output
    return


if __name__ == "__main__":
    app.run()
