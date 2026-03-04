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


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Agentic AI: From Reasoning to Action

        *The ReAct loop, LangChain tools, and context engineering through multi-agent isolation*

        ::: {.callout-note title="What you'll learn in this module"}
        This module introduces agentic AI systems. We will explore how an agent differs from
        a chatbot by operating in a feedback loop, examine the ReAct pattern (Reason and Act)
        that structures this loop, and understand how context engineering through multi-agent
        isolation improves accuracy on complex tasks.
        :::
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Configuration")
    return


@app.cell
def _(mo):
    model_input = mo.ui.text(
        value="ollama/glm4:9b",
        label="Model (litellm format, e.g. ollama/glm4:9b, openai/gpt-4o, anthropic/claude-3-5-sonnet-20241022)",
        full_width=True,
    )
    api_key_input = mo.ui.text(
        value="",
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
        mo.md("*Change any field above and all cells that use the agent will update automatically.*"),
    ])
    config_panel
    return api_base_input, api_key_input, model_input, config_panel


@app.cell
def _(api_base_input, api_key_input, model_input):
    llm_model = model_input.value
    llm_api_key = api_key_input.value or None
    llm_api_base = api_base_input.value or None
    return llm_api_base, llm_api_key, llm_model


@app.cell
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

    mo.md(f"*Using model: `{llm_model}`*")
    return call_llm, litellm


@app.cell
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


@app.cell
def _(mo, plt):
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Chatbot diagram
    _ax1.set_xlim(0, 10)
    _ax1.set_ylim(0, 6)
    _ax1.axis("off")
    _ax1.set_title("Chatbot: Input → Output", fontsize=12, fontweight="bold")
    _ax1.add_patch(plt.Rectangle((0.5, 2.5), 2.5, 1.5, fill=True, facecolor="#e3f2fd", edgecolor="#1565c0", linewidth=2))
    _ax1.text(1.75, 3.25, "Input", ha="center", va="center", fontsize=11)
    _ax1.annotate("", xy=(6.5, 3.25), xytext=(3.5, 3.25),
                  arrowprops=dict(arrowstyle="->", color="#1565c0", lw=2))
    _ax1.add_patch(plt.Rectangle((6.5, 2.5), 2.5, 1.5, fill=True, facecolor="#e8f5e9", edgecolor="#2e7d32", linewidth=2))
    _ax1.text(7.75, 3.25, "Output", ha="center", va="center", fontsize=11)

    # Agent diagram
    _ax2.set_xlim(0, 10)
    _ax2.set_ylim(0, 8)
    _ax2.axis("off")
    _ax2.set_title("Agent: Observe → Think → Act → Observe", fontsize=12, fontweight="bold")
    stages = [("Observe", 5, 6.5, "#fff9c4", "#f9a825"),
              ("Think", 8, 4, "#fce4ec", "#c62828"),
              ("Act", 5, 1.5, "#e8f5e9", "#2e7d32"),
              ("Observe", 2, 4, "#e3f2fd", "#1565c0")]
    for _label, _x, _y, _fc, _ec in stages:
        _ax2.add_patch(plt.Rectangle((_x - 1.2, _y - 0.6), 2.4, 1.2,
                                      fill=True, facecolor=_fc, edgecolor=_ec, linewidth=2))
        _ax2.text(_x, _y, _label, ha="center", va="center", fontsize=10, fontweight="bold")

    arrows = [(5, 6.5 - 0.6, 8, 4 + 0.6), (8, 4 - 0.6, 5, 1.5 + 0.6),
              (5, 1.5 - 0.6, 2, 4 - 0.6), (2, 4 + 0.6, 5, 6.5 - 0.6)]
    for _x1, _y1, _x2, _y2 in arrows:
        _ax2.annotate("", xy=(_x2, _y2), xytext=(_x1, _y1),
                      arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5))

    plt.tight_layout()
    mo.as_html(_fig)
    return FancyArrowPatch, mpatches


@app.cell
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


@app.cell
def _(mo):
    mo.vstack([
        mo.callout(
            mo.md(
                """```
Observation: [initial question or tool result]
Thought:     [chain-of-thought reasoning step]
Action:      [tool_name]
Action Input: [tool parameters as JSON]
Observation: [tool return value]
Thought:     [reasoning based on observation]
... (repeat until done)
Thought:     I now have enough information to answer.
Final Answer: [the answer to the original question]
```"""
            ),
            kind="info",
        ),
        mo.md(
            "::: {.column-margin}\n"
            "**Reference:** Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* "
            "ICLR 2023. The paper showed that interleaving reasoning traces with actions outperforms "
            "reasoning-only (CoT) and acting-only (tool use) baselines on knowledge-intensive tasks.\n"
            ":::"
        ),
    ])
    return


@app.cell
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


@app.cell
def _(mo):
    import datetime

    def get_current_date() -> str:
        """Return today's date as a string in YYYY-MM-DD format.
        Use this tool when the user asks what today's date is or needs the current date for a calculation."""
        return datetime.date.today().isoformat()

    def evaluate_math(expression: str) -> str:
        """Evaluate a mathematical expression and return the result as a string.
        The expression must be a valid Python arithmetic expression (e.g., '2 + 2', '3 * (4 + 5)', '10 / 3').
        Do not use this tool for symbolic algebra. Use it only for numeric calculations."""
        try:
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression):
                return "Error: expression contains disallowed characters."
            result = eval(expression, {"__builtins__": {}})  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def define_word(word: str) -> str:
        """Look up the definition of a common English word.
        Returns a brief one-sentence definition. Use this when the user asks what a word means.
        This tool covers only common dictionary words; it cannot look up technical jargon or proper nouns."""
        definitions = {
            "serendipity": "The occurrence of fortunate events by chance rather than design.",
            "ephemeral": "Lasting for a very short time; transitory.",
            "algorithm": "A step-by-step procedure for solving a problem or accomplishing a task.",
            "entropy": "A measure of disorder or randomness in a system.",
            "heuristic": "A practical problem-solving approach that is not guaranteed to be perfect but is sufficient for the immediate goal.",
        }
        return definitions.get(word.lower(), f"Definition not found for '{word}'.")

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
    return datetime, define_word, evaluate_math, get_current_date


@app.cell
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


@app.cell
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


@app.cell
def _(
    agent_question,
    call_llm,
    define_word,
    evaluate_math,
    get_current_date,
    mo,
    run_agent_btn,
):
    import json
    import re as _re

    TOOLS = {
        "get_current_date": get_current_date,
        "evaluate_math": evaluate_math,
        "define_word": define_word,
    }

    TOOL_DESCRIPTIONS = (
        "You have access to three tools:\n"
        "1. get_current_date() — Return today's date in YYYY-MM-DD format.\n"
        "2. evaluate_math(expression: str) — Evaluate a Python arithmetic expression.\n"
        "3. define_word(word: str) — Look up the definition of a common English word.\n\n"
        "Use this format:\n"
        "Thought: [your reasoning]\n"
        "Action: [tool name]\n"
        "Action Input: [input as a plain string]\n"
        "Observation: [tool result — filled in by the system]\n"
        "... repeat as needed ...\n"
        "Thought: I now have the answer.\n"
        "Final Answer: [your final answer]"
    )

    def run_react_agent(question: str, max_steps: int = 8) -> list:
        """Run a simple ReAct loop and return the trace as a list of dicts."""
        trace = []
        messages = [
            {"role": "system", "content": TOOL_DESCRIPTIONS},
            {"role": "user", "content": question},
        ]

        for _step in range(max_steps):
            try:
                response = call_llm(messages)
                text = response.choices[0].message.content
            except Exception as e:
                trace.append({"type": "error", "content": str(e)})
                break

            messages.append({"role": "assistant", "content": text})

            if "Final Answer:" in text:
                _fa_idx = text.index("Final Answer:")
                trace.append({"type": "final", "content": text[_fa_idx + len("Final Answer:"):].strip()})
                break

            action_match = _re.search(r"Action:\s*(.+)", text)
            input_match = _re.search(r"Action Input:\s*(.+)", text)
            thought_match = _re.search(r"Thought:\s*(.+?)(?:\nAction|$)", text, _re.DOTALL)

            if thought_match:
                trace.append({"type": "thought", "content": thought_match.group(1).strip()})

            if action_match and input_match:
                tool_name = action_match.group(1).strip()
                tool_input = input_match.group(1).strip()

                trace.append({"type": "action", "tool": tool_name, "input": tool_input})

                if tool_name in TOOLS:
                    try:
                        obs = TOOLS[tool_name](tool_input) if tool_input else TOOLS[tool_name]()
                    except TypeError:
                        try:
                            obs = TOOLS[tool_name]()
                        except Exception as e:
                            obs = f"Error: {e}"
                else:
                    obs = f"Tool '{tool_name}' not found. Available: {list(TOOLS.keys())}"

                trace.append({"type": "observation", "content": obs})
                messages.append({"role": "user", "content": f"Observation: {obs}"})
            else:
                trace.append({"type": "thought", "content": text})
                break

        return trace

    if run_agent_btn.value:
        _trace = run_react_agent(agent_question.value)
        _trace_parts = [mo.md(f"### ReAct Trace ({len(_trace)} steps)")]
        for _step_item in _trace:
            if _step_item["type"] == "thought":
                _trace_parts.append(mo.callout(mo.md(f"**Thought:** {_step_item['content']}"), kind="info"))
            elif _step_item["type"] == "action":
                _trace_parts.append(mo.callout(
                    mo.md(f"**Action:** `{_step_item['tool']}`\n\n**Input:** `{_step_item['input']}`"),
                    kind="warn",
                ))
            elif _step_item["type"] == "observation":
                _trace_parts.append(mo.callout(mo.md(f"**Observation:** {_step_item['content']}"), kind="neutral"))
            elif _step_item["type"] == "final":
                _trace_parts.append(mo.callout(mo.md(f"**Final Answer:** {_step_item['content']}"), kind="success"))
            elif _step_item["type"] == "error":
                _trace_parts.append(mo.callout(mo.md(f"**Error:** {_step_item['content']}"), kind="danger"))
        mo.vstack(_trace_parts)
    else:
        mo.md("*Click **Run agent** to watch the ReAct loop unfold.*")
    return TOOL_DESCRIPTIONS, TOOLS, json, run_react_agent


@app.cell
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


@app.cell
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

    mo.md(f"*Titanic dataset loaded: {len(titanic_df)} rows, {len(titanic_df.columns)} columns.*")
    return duckdb, pd, titanic_df


@app.cell
def _(duckdb, mo, titanic_df):
    def inspect_schema() -> str:
        """Return the column names and data types of the Titanic dataset.
        Use this tool first to understand what columns are available before running any query."""
        return titanic_df.dtypes.to_string()

    def get_sample_rows(n: str = "5") -> str:
        """Return the first n rows of the Titanic dataset as a string.
        Use this to get a concrete sense of what the data looks like.
        The parameter n must be a string representation of an integer (e.g., '5', '10')."""
        try:
            return titanic_df.head(int(n)).to_string()
        except Exception as e:
            return f"Error: {e}"

    def run_sql_query(query: str) -> str:
        """Run a SQL query against the Titanic dataset and return the result as a string.
        The table is named 'titanic_df'. Write standard SQL (SELECT, WHERE, GROUP BY, ORDER BY).
        Example: SELECT AVG(Age) FROM titanic_df WHERE Survived = 1
        Always check inspect_schema first to confirm column names."""
        try:
            result = duckdb.query(query).df()
            return result.to_string(index=False)
        except Exception as e:
            return f"SQL Error: {e}. Available columns: {list(titanic_df.columns)}"

    def get_summary_stats(column: str) -> str:
        """Return descriptive statistics (mean, std, min, max, quartiles) for a numeric column.
        The column parameter must be an exact column name from the Titanic dataset (case-sensitive).
        Use inspect_schema first to confirm the exact column name."""
        if column not in titanic_df.columns:
            return f"Column '{column}' not found. Available columns: {list(titanic_df.columns)}"
        return titanic_df[column].describe().to_string()

    TITANIC_TOOLS = {
        "inspect_schema": lambda _: inspect_schema(),
        "get_sample_rows": get_sample_rows,
        "run_sql_query": run_sql_query,
        "get_summary_stats": get_summary_stats,
    }

    TITANIC_SYSTEM_PROMPT = (
        "You are a data detective analyzing the Titanic dataset. "
        "You MUST verify every claim with at least one tool call before giving your Final Answer. "
        "Never guess — always query the data.\n\n"
        "Available tools:\n"
        "1. inspect_schema() — Get column names and types.\n"
        "2. get_sample_rows(n) — Get the first n rows.\n"
        "3. run_sql_query(query) — Run SQL against the Titanic dataset (table: titanic_df).\n"
        "4. get_summary_stats(column) — Get descriptive statistics for a column.\n\n"
        "Use this format:\n"
        "Thought: [reasoning]\nAction: [tool name]\nAction Input: [input]\n"
        "Observation: [filled by system]\n... repeat ...\nFinal Answer: [answer]"
    )

    mo.md("*Titanic tools defined and ready.*")
    return (
        TITANIC_SYSTEM_PROMPT,
        TITANIC_TOOLS,
        get_sample_rows,
        get_summary_stats,
        inspect_schema,
        run_sql_query,
    )


@app.cell
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


@app.cell
def _(
    TITANIC_SYSTEM_PROMPT,
    TITANIC_TOOLS,
    call_llm,
    mo,
    preset_q1,
    preset_q2,
    preset_q3,
    _re,
    run_titanic_btn,
    titanic_question,
):
    def run_titanic_agent(question: str, max_steps: int = 10) -> list:
        """Run the Titanic data detective agent."""
        trace = []
        messages = [
            {"role": "system", "content": TITANIC_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        for _ in range(max_steps):
            try:
                resp = call_llm(messages)
                text = resp.choices[0].message.content
            except Exception as e:
                trace.append({"type": "error", "content": str(e)})
                break

            messages.append({"role": "assistant", "content": text})

            if "Final Answer:" in text:
                fa_idx = text.index("Final Answer:")
                trace.append({"type": "final", "content": text[fa_idx + 13:].strip()})
                break

            action_m = _re.search(r"Action:\s*(.+)", text)
            input_m = _re.search(r"Action Input:\s*(.+)", text, _re.DOTALL)
            thought_m = _re.search(r"Thought:\s*(.+?)(?:\nAction|$)", text, _re.DOTALL)

            if thought_m:
                trace.append({"type": "thought", "content": thought_m.group(1).strip()})

            if action_m and input_m:
                tool_name = action_m.group(1).strip()
                tool_input = input_m.group(1).strip().split("\n")[0].strip()
                trace.append({"type": "action", "tool": tool_name, "input": tool_input})

                if tool_name in TITANIC_TOOLS:
                    try:
                        obs = TITANIC_TOOLS[tool_name](tool_input)
                    except Exception as e:
                        obs = f"Error: {e}"
                else:
                    obs = f"Tool '{tool_name}' not found. Available: {list(TITANIC_TOOLS.keys())}"

                trace.append({"type": "observation", "content": obs})
                messages.append({"role": "user", "content": f"Observation: {obs}"})
            else:
                trace.append({"type": "thought", "content": text})
                break

        return trace

    _q = None
    if preset_q1.value:
        _q = "What was the overall survival rate?"
    elif preset_q2.value:
        _q = "What is the average age of survivors vs. non-survivors?"
    elif preset_q3.value:
        _q = "Which embarkation port had the highest survival rate?"
    elif run_titanic_btn.value:
        _q = titanic_question.value

    if _q:
        _trace = run_titanic_agent(_q)
        _parts = [mo.md(f"### Titanic Agent Trace — *{_q}*")]
        for _item in _trace:
            if _item["type"] == "thought":
                _parts.append(mo.callout(mo.md(f"**Thought:** {_item['content']}"), kind="info"))
            elif _item["type"] == "action":
                _parts.append(mo.callout(
                    mo.md(f"**Action:** `{_item['tool']}`\n\n**Input:** `{_item['input']}`"),
                    kind="warn",
                ))
            elif _item["type"] == "observation":
                _parts.append(mo.callout(mo.md(f"**Observation:**\n```\n{_item['content']}\n```"), kind="neutral"))
            elif _item["type"] == "final":
                _parts.append(mo.callout(mo.md(f"**Final Answer:** {_item['content']}"), kind="success"))
            elif _item["type"] == "error":
                _parts.append(mo.callout(mo.md(f"**Error:** {_item['content']}"), kind="danger"))
        mo.vstack(_parts)
    else:
        mo.md("*Select a preset question or type your own and click **Run data detective**.*")
    return run_titanic_agent,


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Section 6: Context Engineering — Multi-Agent Isolation for Literature Verification

        When you give a single agent a long list of things to verify, the references start to
        bleed into each other. The agent may use a detail from one paper to justify a claim
        about another. This is context confusion, and it grows worse as the context window fills.

        Below are five academic references: three real, two fabricated. First, watch a monolithic
        agent receive all five at once and assess each one. Then watch five isolated sub-agents,
        each receiving only its single reference. Compare the accuracy and the token counts.
        """
    )
    return


@app.cell
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


@app.cell
def _(REFERENCES, call_llm, mo, run_monolithic_btn, run_multiagent_btn):
    import asyncio

    VERIFY_SYSTEM = (
        "You are a fact-checker for academic references. "
        "For each reference, state whether it is REAL or FABRICATED and give a brief reason. "
        "Be concise."
    )

    async def verify_single(ref: dict) -> dict:
        """Verify one reference in isolation."""
        messages = [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": f"Verify this reference: {ref['citation']}"},
        ]
        try:
            resp = call_llm(messages)
            return {"id": ref["id"], "verdict": resp.choices[0].message.content, "real": ref["real"]}
        except Exception as e:
            return {"id": ref["id"], "verdict": f"Error: {e}", "real": ref["real"]}

    if run_monolithic_btn.value:
        _all_refs = "\n".join(f"[{r['id']}] {r['citation']}" for r in REFERENCES)
        _messages = [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": f"Verify each of these five references:\n\n{_all_refs}"},
        ]
        try:
            _resp = call_llm(_messages)
            _mono_text = _resp.choices[0].message.content
            mo.vstack([
                mo.md("### Monolithic agent response"),
                mo.callout(mo.md(_mono_text), kind="info"),
                mo.md("**Ground truth:**"),
                mo.md("\n\n".join(f"**{r['id']}:** {r['verdict']}" for r in REFERENCES)),
            ])
        except Exception as _e:
            mo.callout(mo.md(f"**Error:** {_e}"), kind="danger")

    elif run_multiagent_btn.value:
        try:
            _results = asyncio.get_event_loop().run_until_complete(
                asyncio.gather(*[verify_single(r) for r in REFERENCES])
            )
            _parts = [mo.md("### Multi-agent results (one agent per reference)")]
            for _r in _results:
                _gt = next(ref for ref in REFERENCES if ref["id"] == _r["id"])
                _parts.append(mo.vstack([
                    mo.md(f"**{_r['id']}** ({'✅ Real' if _gt['real'] else '❌ Fabricated'} — ground truth)"),
                    mo.callout(mo.md(_r["verdict"]), kind="info"),
                ]))
            _parts.append(mo.md("**Ground truth:**"))
            _parts.append(mo.md("\n\n".join(f"**{r['id']}:** {r['verdict']}" for r in REFERENCES)))
            mo.vstack(_parts)
        except Exception as _e:
            mo.callout(mo.md(f"**Error:** {_e}"), kind="danger")
    else:
        mo.md("*Click a button above to run the monolithic or multi-agent demo.*")
    return VERIFY_SYSTEM, asyncio, verify_single


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Student Task: Extend the Data Detective

        ::: {.callout-tip title="Try it yourself"}
        Add two new tools to the Titanic agent. The first returns a cross-tabulation (pivot table)
        of two columns. The second filters rows by a column value and returns a count.

        After adding both tools, pose this question to the agent: "Among female passengers over
        30 years old, which ticket class had the highest survival rate, and how does it compare
        to males in the same age group?"

        The agent must chain at least three tool calls to answer correctly. A ground-truth cell
        verifies the numerical answer.

        **Hint:** Write the docstring first. Describe exactly what the function returns, what
        the parameters mean, and what format the output is in. The LLM will call your tool
        exactly as described — be precise.

        **Extension:** Modify the system prompt to instruct the agent to always question its
        first answer and run one additional verification tool call. Does this improve accuracy?
        :::
        """
    )
    return


@app.cell
def _(mo, titanic_df):
    mo.md("### Your tools go here — edit this cell to add cross_tabulation and filter_and_count")

    def cross_tabulation(col1_col2: str) -> str:
        """Return a cross-tabulation of two columns in the Titanic dataset as a string.
        The input must be two column names separated by a comma, e.g., 'Sex,Survived'.
        The first column becomes the rows, the second becomes the columns.
        Use this to compare counts across two categorical variables."""
        try:
            cols = [c.strip() for c in col1_col2.split(",")]
            if len(cols) != 2:
                return "Error: provide exactly two column names separated by a comma."
            return pd.crosstab(titanic_df[cols[0]], titanic_df[cols[1]]).to_string()
        except Exception as e:
            return f"Error: {e}. Available columns: {list(titanic_df.columns)}"

    def filter_and_count(condition: str) -> str:
        """Filter the Titanic dataset by a condition and return the count of matching rows.
        The condition must be a valid pandas query string, e.g., 'Sex == \"female\" and Age > 30'.
        Use this to answer 'how many passengers' questions with specific criteria.
        Always check inspect_schema first to confirm column names and data types."""
        try:
            count = len(titanic_df.query(condition))
            return f"{count} rows match the condition: {condition}"
        except Exception as e:
            return f"Error: {e}. Available columns: {list(titanic_df.columns)}"

    mo.md(
        "Two starter tools are provided above (`cross_tabulation` and `filter_and_count`). "
        "Review their docstrings, then run the agent below with the target question."
    )
    return cross_tabulation, filter_and_count


@app.cell
def _(mo):
    import pandas as pd  # noqa: F811 — needed for cross_tabulation cell
    return (pd,)


@app.cell
def _(
    TITANIC_SYSTEM_PROMPT,
    TITANIC_TOOLS,
    call_llm,
    cross_tabulation,
    filter_and_count,
    get_sample_rows,
    get_summary_stats,
    inspect_schema,
    mo,
    _re,
    run_sql_query,
    run_titanic_agent,
    titanic_df,
):
    EXTENDED_TOOLS = dict(TITANIC_TOOLS)
    EXTENDED_TOOLS["cross_tabulation"] = cross_tabulation
    EXTENDED_TOOLS["filter_and_count"] = filter_and_count

    TARGET_QUESTION = (
        "Among female passengers over 30 years old, which ticket class had the highest survival rate, "
        "and how does it compare to males in the same age group?"
    )

    run_extended_btn = mo.ui.run_button(label="Run extended agent with target question")
    run_extended_btn
    return EXTENDED_TOOLS, TARGET_QUESTION, run_extended_btn


@app.cell
def _(
    EXTENDED_TOOLS,
    TITANIC_SYSTEM_PROMPT,
    TARGET_QUESTION,
    call_llm,
    mo,
    _re,
    run_extended_btn,
    titanic_df,
):
    def run_extended_agent(question: str, max_steps: int = 12) -> list:
        trace = []
        ext_system = (
            TITANIC_SYSTEM_PROMPT +
            "\n5. cross_tabulation(col1,col2) — Cross-tabulate two columns.\n"
            "6. filter_and_count(condition) — Count rows matching a pandas query condition."
        )
        messages = [
            {"role": "system", "content": ext_system},
            {"role": "user", "content": question},
        ]
        for _ in range(max_steps):
            try:
                resp = call_llm(messages)
                text = resp.choices[0].message.content
            except Exception as e:
                trace.append({"type": "error", "content": str(e)})
                break

            messages.append({"role": "assistant", "content": text})

            if "Final Answer:" in text:
                fa_idx = text.index("Final Answer:")
                trace.append({"type": "final", "content": text[fa_idx + 13:].strip()})
                break

            action_m = _re.search(r"Action:\s*(.+)", text)
            input_m = _re.search(r"Action Input:\s*(.+)", text, _re.DOTALL)
            thought_m = _re.search(r"Thought:\s*(.+?)(?:\nAction|$)", text, _re.DOTALL)

            if thought_m:
                trace.append({"type": "thought", "content": thought_m.group(1).strip()})

            if action_m and input_m:
                tool_name = action_m.group(1).strip()
                tool_input = input_m.group(1).strip().split("\n")[0].strip()
                trace.append({"type": "action", "tool": tool_name, "input": tool_input})

                if tool_name in EXTENDED_TOOLS:
                    try:
                        obs = EXTENDED_TOOLS[tool_name](tool_input)
                    except Exception as e:
                        obs = f"Error: {e}"
                else:
                    obs = f"Tool '{tool_name}' not found. Available: {list(EXTENDED_TOOLS.keys())}"

                trace.append({"type": "observation", "content": obs})
                messages.append({"role": "user", "content": f"Observation: {obs}"})
            else:
                trace.append({"type": "thought", "content": text})
                break

        return trace

    # Ground truth computation
    _female_over30 = titanic_df[(titanic_df["Sex"] == "female") & (titanic_df["Age"] > 30)]
    _male_over30 = titanic_df[(titanic_df["Sex"] == "male") & (titanic_df["Age"] > 30)]
    _f_rates = _female_over30.groupby("Pclass")["Survived"].mean()
    _m_rates = _male_over30.groupby("Pclass")["Survived"].mean()
    _best_f_class = int(_f_rates.idxmax())
    _best_f_rate = float(_f_rates.max())
    _best_m_class = int(_m_rates.idxmax())
    _best_m_rate = float(_m_rates.max())

    if run_extended_btn.value:
        _etrace = run_extended_agent(TARGET_QUESTION)
        _eparts = [mo.md("### Extended Agent Trace")]
        for _item in _etrace:
            if _item["type"] == "thought":
                _eparts.append(mo.callout(mo.md(f"**Thought:** {_item['content']}"), kind="info"))
            elif _item["type"] == "action":
                _eparts.append(mo.callout(
                    mo.md(f"**Action:** `{_item['tool']}`\n\n**Input:** `{_item['input']}`"),
                    kind="warn",
                ))
            elif _item["type"] == "observation":
                _eparts.append(mo.callout(mo.md(f"**Observation:**\n```\n{_item['content']}\n```"), kind="neutral"))
            elif _item["type"] == "final":
                _eparts.append(mo.callout(mo.md(f"**Final Answer:** {_item['content']}"), kind="success"))
            elif _item["type"] == "error":
                _eparts.append(mo.callout(mo.md(f"**Error:** {_item['content']}"), kind="danger"))

        _eparts.append(mo.callout(
            mo.md(
                f"**Ground truth:**\n\n"
                f"Female passengers over 30: Class {_best_f_class} had highest survival rate ({_best_f_rate:.1%}).\n\n"
                f"Male passengers over 30: Class {_best_m_class} had highest survival rate ({_best_m_rate:.1%})."
            ),
            kind="neutral",
        ))
        mo.vstack(_eparts)
    else:
        mo.callout(
            mo.md(
                f"**Ground truth (revealed after you run):**\n\n"
                f"Female passengers over 30: Class {_best_f_class} had highest survival rate ({_best_f_rate:.1%}).\n\n"
                f"Male passengers over 30: Class {_best_m_class} had highest survival rate ({_best_m_rate:.1%})."
            ),
            kind="neutral",
        )
    return run_extended_agent,


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Thought Experiment: When Does the Loop Break?

        ::: {.callout-tip title="Try it yourself"}
        A broken query tool is defined below that always raises a ValueError with a vague
        error message. Run the agent on a simple question and watch what happens. Does the
        agent retry? Does it give up? Does it hallucinate an answer after failing?

        Then edit the error message to be more informative: name the available columns and
        explain what went wrong. Run the agent again and compare.

        **Hint:** Agents learn from their observations. A tool that says "Error" is less
        useful than a tool that says "ValueError: column 'Age' not found. Available columns:
        Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked."
        :::
        """
    )
    return


@app.cell
def _(
    TITANIC_TOOLS,
    call_llm,
    mo,
    _re,
    titanic_df,
):
    def broken_query_tool(query: str) -> str:
        """Run a SQL query against the Titanic dataset.
        IMPORTANT: Edit the error message below to make it more informative."""
        raise ValueError("Error")  # TODO: make this error message more helpful

    BROKEN_TOOLS = dict(TITANIC_TOOLS)
    BROKEN_TOOLS["run_sql_query"] = broken_query_tool

    BROKEN_QUESTION = "What is the average fare paid by first-class passengers?"

    run_broken_btn = mo.ui.run_button(label="Run agent with broken tool")

    def run_broken_agent(question: str, max_steps: int = 6) -> list:
        trace = []
        messages = [
            {"role": "system", "content": (
                "You are a data analyst. Use run_sql_query(query) to answer questions about "
                "the Titanic dataset (table: titanic_df). Always query before answering.\n\n"
                "Format:\nThought: [reason]\nAction: run_sql_query\nAction Input: [SQL]\n"
                "Observation: [result]\n...\nFinal Answer: [answer]"
            )},
            {"role": "user", "content": question},
        ]
        for _ in range(max_steps):
            try:
                resp = call_llm(messages)
                text = resp.choices[0].message.content
            except Exception as e:
                trace.append({"type": "error", "content": str(e)})
                break

            messages.append({"role": "assistant", "content": text})

            if "Final Answer:" in text:
                fa_idx = text.index("Final Answer:")
                trace.append({"type": "final", "content": text[fa_idx + 13:].strip()})
                break

            action_m = _re.search(r"Action:\s*(.+)", text)
            input_m = _re.search(r"Action Input:\s*(.+)", text, _re.DOTALL)
            thought_m = _re.search(r"Thought:\s*(.+?)(?:\nAction|$)", text, _re.DOTALL)

            if thought_m:
                trace.append({"type": "thought", "content": thought_m.group(1).strip()})

            if action_m and input_m:
                tool_name = action_m.group(1).strip()
                tool_input = input_m.group(1).strip().split("\n")[0]
                trace.append({"type": "action", "tool": tool_name, "input": tool_input})

                fn = BROKEN_TOOLS.get(tool_name)
                if fn:
                    try:
                        obs = fn(tool_input)
                    except Exception as e:
                        obs = str(e)
                else:
                    obs = f"Tool '{tool_name}' not found."

                trace.append({"type": "observation", "content": obs})
                messages.append({"role": "user", "content": f"Observation: {obs}"})
            else:
                trace.append({"type": "thought", "content": text})
                break

        return trace

    mo.vstack([
        mo.callout(mo.md(f"**Question:** {BROKEN_QUESTION}"), kind="info"),
        run_broken_btn,
    ])
    return BROKEN_QUESTION, BROKEN_TOOLS, run_broken_agent, run_broken_btn, broken_query_tool


@app.cell
def _(BROKEN_QUESTION, mo, run_broken_agent, run_broken_btn):
    if run_broken_btn.value:
        _btrace = run_broken_agent(BROKEN_QUESTION)
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
        mo.vstack(_bparts)
    else:
        mo.md("*Click **Run agent with broken tool** to watch what happens.*")
    return


@app.cell
def _(mo):
    import matplotlib.pyplot as plt
    return (plt,)


if __name__ == "__main__":
    app.run()
