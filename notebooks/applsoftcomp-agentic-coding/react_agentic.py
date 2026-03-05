# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "litellm==1.82.0",
#   "pandas==3.0.1",
#   "matplotlib==3.10.8",
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
def _():
    import re
    import matplotlib.pyplot as plt
    return plt, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Agentic AI: Tools, Loops, and Conversations

    *Build agents that reason and act — then add your own tool*

    /// tip | How to run this notebook
    Download this file, then open a terminal and run:

    ```
    marimo edit --sandbox react_agentic.py
    ```

    If you do not have marimo installed, run it without installation using
    `uvx marimo edit --sandbox react_agentic.py`. The `--sandbox` flag creates
    an isolated environment and installs all dependencies automatically.
    ///

    /// note | What you'll learn in this module
    This module introduces agentic AI systems. We will see how tools turn a
    language model into an agent that can act on the world, explore two complete
    agents through a natural chat interface, and build a visualization agent that
    draws Titanic dataset plots on request. By the end, you will define your own
    tool and talk to an agent that uses it.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Configuration")
    return


@app.cell(hide_code=True)
def _(mo):
    model_input = mo.ui.text(
        value="ollama/ministral-3:14b-cloud",
        label="Model (litellm format, e.g. ollama/glm-4.7:cloud, openai/gpt-4o)",
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
    mo.vstack([
        mo.md("### LLM Configuration"),
        model_input,
        api_key_input,
        api_base_input,
        mo.md(
            "*Change any field above and all agent cells update automatically.*\n\n"
            "**Default model:** `ollama/ministral-3:14b-cloud` — a free cloud model through "
            "your local ollama installation. No API key required. "
            "Run `ollama list` in your terminal to see all available models."
        ),
    ])
    return api_base_input, api_key_input, model_input


@app.cell(hide_code=True)
def _(mo):
    import subprocess as _subprocess
    try:
        _r = _subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        _out = _r.stdout.strip() if _r.returncode == 0 else _r.stderr.strip()
        _ok = _r.returncode == 0
    except FileNotFoundError:
        _out = "ollama not found. Install from https://ollama.com or use a cloud model string."
        _ok = False
    except Exception as _e:
        _out = f"Could not run ollama list: {_e}"
        _ok = False

    if _ok and _out:
        _display = mo.callout(mo.vstack([
            mo.md("**Available ollama models** (from `ollama list`):"),
            mo.md(f"```\n{_out}\n```"),
            mo.md("Use the model name as `ollama/<model-name>` in the config panel above."),
        ]), kind="success")
    elif _ok:
        _display = mo.callout(
            mo.md("No local ollama models found. Run `ollama pull <model-name>` to download one."),
            kind="warn",
        )
    else:
        _display = mo.callout(mo.md(f"**ollama not available:** {_out}"), kind="warn")
    mo.accordion({"Available ollama models (click to expand)": _display})
    return


@app.cell(hide_code=True)
def _(api_base_input, api_key_input, model_input):
    llm_model = model_input.value.strip()
    llm_api_key = api_key_input.value or None
    llm_api_base = api_base_input.value or None
    return llm_api_base, llm_api_key, llm_model


@app.cell(hide_code=True)
def _(llm_api_base, llm_api_key, llm_model, mo):
    import litellm

    def call_llm(messages: list):
        """Call the configured LLM with a list of messages."""
        kwargs = {"model": llm_model, "messages": messages}
        if llm_api_key:
            kwargs["api_key"] = llm_api_key
        if llm_api_base:
            kwargs["api_base"] = llm_api_base
        try:
            return litellm.completion(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"LLM call failed for model `{llm_model}`: {e}\n"
                "Check the model string and API settings in the config panel above."
            )

    mo.accordion({
        "call_llm() helper — click to view": mo.md(
            f"*Using model: `{llm_model}`*\n\n"
            "```python\n"
            "def call_llm(messages: list):\n"
            "    # Routes to the configured LLM via litellm.completion().\n"
            "    # Handles api_key and api_base automatically.\n"
            "    ...\n"
            "```"
        )
    })
    return (call_llm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What Is an Agent?

    A chatbot takes your message and returns a reply. The interaction is a
    single function call: one input, one output, done.

    An agent is different. It operates in a loop. It reads a question, decides
    whether it needs more information, calls a **tool**, reads the result, and
    repeats — until it has enough to answer. The feedback loop is what allows
    an agent to act on the world rather than just recite from memory.

    The key ingredient is tools. A tool is just a Python function. The agent
    sees the function's name and its docstring, and from that description alone
    it decides when to call the function and what arguments to pass. Below we
    will build two agents. The first knows how to look up the current date and
    evaluate arithmetic. The second can draw plots of the Titanic dataset on
    request. The only difference between them is which tools are in the toolbox.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 1: A Simple Agent — Date and Math

    Here are the two tools for the first agent. Notice that the body of each
    function is ordinary Python. The docstring is what the agent reads: it tells
    the model what the tool does, what arguments to pass, and what it returns.
    Write a vague docstring and the agent will call the tool incorrectly. Write
    a precise one and it will use it reliably.
    """)
    return


@app.cell
def _():
    import datetime

    def get_current_date() -> str:
        """Return today's date as YYYY-MM-DD. Takes no arguments."""
        return datetime.date.today().isoformat()

    def evaluate_math(expr: str) -> str:
        """
        Evaluate a Python arithmetic expression and return the result as a string.
        expr: a valid Python math expression such as '2 + 2' or '17 * 42 + 8'.
        Only arithmetic is supported. Do not pass variables or function calls.
        """
        try:
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expr):
                return "Error: expression contains disallowed characters."
            return str(eval(expr, {"__builtins__": {}}))  # noqa: S307
        except Exception as e:
            return f"Error: {e}"

    return datetime, evaluate_math, get_current_date


@app.cell(hide_code=True)
def _(call_llm, evaluate_math, get_current_date, re):
    import inspect as _inspect

    _SIMPLE_TOOLS = {
        "get_current_date": get_current_date,
        "evaluate_math": evaluate_math,
    }

    _SIMPLE_DESC = (
        "You are a helpful assistant with access to exactly two tools.\n\n"
        "Tools:\n"
        "  1. get_current_date() — returns today's date as YYYY-MM-DD. Takes NO arguments.\n"
        "  2. evaluate_math(expr) — evaluates a Python arithmetic expression, e.g. '17 * 42 + 8'.\n\n"
        "Always respond in this exact format:\n"
        "Thought: <your reasoning>\n"
        "Action: <tool_name>\n"
        "Action Input: <argument, or 'none' if the tool takes no arguments>\n\n"
        "After receiving an Observation, continue with another Thought/Action or write:\n"
        "Final Answer: <your final answer to the user>"
    )

    def run_simple_agent(question: str, max_steps: int = 8) -> list:
        msgs = [{"role": "system", "content": _SIMPLE_DESC}, {"role": "user", "content": question}]
        trace = []
        for _ in range(max_steps):
            try:
                text = call_llm(msgs).choices[0].message.content
            except Exception as e:
                trace.append({"type": "error", "content": str(e)})
                break
            msgs.append({"role": "assistant", "content": text})
            if "Final Answer:" in text:
                trace.append({"type": "final", "content": text[text.index("Final Answer:") + 13:].strip()})
                break
            if tm := re.search(r"Thought:\s*(.+?)(?:\nAction|$)", text, re.DOTALL):
                trace.append({"type": "thought", "content": tm.group(1).strip()})
            am = re.search(r"Action:\s*(.+)", text)
            im = re.search(r"Action Input:\s*(.+)", text)
            if am:
                tn = am.group(1).strip()
                ti = im.group(1).strip() if im else "none"
                trace.append({"type": "action", "tool": tn, "input": ti})
                if tn in _SIMPLE_TOOLS:
                    try:
                        params = _inspect.signature(_SIMPLE_TOOLS[tn]).parameters
                        obs = _SIMPLE_TOOLS[tn]() if not params else _SIMPLE_TOOLS[tn](ti)
                    except Exception as e:
                        obs = f"Error: {e}"
                else:
                    obs = f"Tool '{tn}' not found. Available: {', '.join(_SIMPLE_TOOLS)}."
                trace.append({"type": "observation", "content": str(obs)})
                msgs.append({"role": "user", "content": f"Observation: {obs}"})
            else:
                trace.append({"type": "thought", "content": text})
                break
        return trace

    return (run_simple_agent,)


@app.cell(hide_code=True)
def _(mo, run_simple_agent):
    def _simple_model(messages, config):
        trace = run_simple_agent(str(messages[-1].content))
        parts = []
        for step in trace:
            if step["type"] == "thought":
                parts.append(mo.md(f"> **Thought:** {step['content']}"))
            elif step["type"] == "action":
                parts.append(mo.md(f"> **Action:** `{step['tool']}` — input: `{step['input']}`"))
            elif step["type"] == "observation":
                parts.append(mo.md(f"> **Observation:** {step['content']}"))
            elif step["type"] == "final":
                parts.append(mo.md(f"\n**Answer:** {step['content']}"))
            elif step["type"] == "error":
                parts.append(mo.md(f"> **Error:** {step['content']}"))
        return mo.vstack(parts) if parts else mo.md("*(No response — check the model config above.)*")

    simple_chat = mo.ui.chat(
        _simple_model,
        prompts=[
            "What is today's date?",
            "What is 17 * 42 + 8?",
            "What is today's date, and how many days are in that year?",
            "Calculate (123 + 456) * 789",
        ],
        show_configuration_controls=False,
    )
    simple_chat
    return (simple_chat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each reply shows the agent's internal reasoning as blockquotes before the
    final answer. That reasoning trace is exactly the ReAct loop at work: Thought
    leads to an Action, the Action returns an Observation, and the Observation
    informs the next Thought. Try asking something that requires two tool calls —
    for instance, today's date combined with an arithmetic question — and watch
    the agent chain the steps together.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 2: A Visualization Agent

    Now let us give the agent a more interesting toolbox. The Titanic dataset is
    loaded below and three visualization tools are available: one that lists the
    column names, one that draws a histogram of any column, and one that draws a
    scatter plot of two columns.

    The tools are defined in the exposed cell below so you can read them the same
    way the agent does — through their docstrings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import pandas as _pd
    try:
        titanic_df = _pd.read_csv(
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        )
        mo.callout(
            mo.md(f"Titanic dataset loaded: **{len(titanic_df)} rows**, **{len(titanic_df.columns)} columns**."),
            kind="success",
        )
    except Exception as _e:
        titanic_df = _pd.DataFrame()
        mo.callout(mo.md(f"Could not load Titanic dataset: {_e}"), kind="danger")
    return (titanic_df,)


@app.cell
def _(mo, plt, titanic_df):
    def list_columns() -> str:
        """Return the names of all columns in the Titanic dataset. Takes no arguments."""
        return ", ".join(titanic_df.columns.tolist())

    def plot_histogram(column: str):
        """
        Draw a histogram of one column from the Titanic dataset and return an image.
        column: the exact column name to plot, e.g. 'Age', 'Fare', 'Pclass', 'Survived'.
        Use list_columns() first if you are unsure of the column names.
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        titanic_df[column].dropna().hist(ax=ax, bins=20, color="#1565c0", edgecolor="white")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {column}")
        plt.tight_layout()
        img = mo.as_html(fig)
        plt.close(fig)
        return img

    def plot_scatter(x_col: str, y_col: str):
        """
        Draw a scatter plot of two columns from the Titanic dataset and return an image.
        x_col: column name for the x-axis, e.g. 'Age'.
        y_col: column name for the y-axis, e.g. 'Fare'.
        Use list_columns() first if you are unsure of the column names.
        When calling this tool, provide both columns separated by a comma: 'Age, Fare'.
        """
        data = titanic_df[[x_col, y_col]].dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(data[x_col], data[y_col], alpha=0.4, color="#1565c0", s=20)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")
        plt.tight_layout()
        img = mo.as_html(fig)
        plt.close(fig)
        return img

    return list_columns, plot_histogram, plot_scatter


@app.cell(hide_code=True)
def _(call_llm, list_columns, mo, plot_histogram, plot_scatter, re):
    import inspect as _inspect2

    _VIZ_TOOLS = {
        "list_columns": list_columns,
        "plot_histogram": plot_histogram,
        "plot_scatter": plot_scatter,
    }

    _VIZ_DESC = (
        "You are a data visualization assistant for the Titanic dataset.\n\n"
        "Tools:\n"
        "  1. list_columns() — returns all column names. Takes NO arguments.\n"
        "  2. plot_histogram(column) — draws a histogram of the given column. Returns an image.\n"
        "     column: the exact column name, e.g. 'Age', 'Fare', 'Pclass'.\n"
        "  3. plot_scatter(x_col, y_col) — draws a scatter plot. Returns an image.\n"
        "     Provide both column names separated by a comma, e.g. 'Age, Fare'.\n\n"
        "Always respond in this exact format:\n"
        "Thought: <your reasoning>\n"
        "Action: <tool_name>\n"
        "Action Input: <argument(s), or 'none' if the tool takes no arguments>\n\n"
        "After receiving an Observation, continue with another Thought/Action or write:\n"
        "Final Answer: <your final answer to the user>"
    )

    def _run_viz_agent(question: str, max_steps: int = 8) -> list:
        msgs = [{"role": "system", "content": _VIZ_DESC}, {"role": "user", "content": question}]
        trace = []
        for _ in range(max_steps):
            try:
                text = call_llm(msgs).choices[0].message.content
            except Exception as e:
                trace.append({"type": "error", "content": str(e)})
                break
            msgs.append({"role": "assistant", "content": text})
            if "Final Answer:" in text:
                trace.append({"type": "final", "content": text[text.index("Final Answer:") + 13:].strip()})
                break
            if tm := re.search(r"Thought:\s*(.+?)(?:\nAction|$)", text, re.DOTALL):
                trace.append({"type": "thought", "content": tm.group(1).strip()})
            am = re.search(r"Action:\s*(.+)", text)
            im = re.search(r"Action Input:\s*(.+)", text)
            if am:
                tn = am.group(1).strip()
                ti = im.group(1).strip() if im else "none"
                trace.append({"type": "action", "tool": tn, "input": ti})
                if tn in _VIZ_TOOLS:
                    try:
                        params = list(_inspect2.signature(_VIZ_TOOLS[tn]).parameters)
                        if not params:
                            result = _VIZ_TOOLS[tn]()
                        elif len(params) == 1:
                            result = _VIZ_TOOLS[tn](ti)
                        else:
                            args = [a.strip() for a in ti.split(",", maxsplit=len(params) - 1)]
                            result = _VIZ_TOOLS[tn](*args)
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = f"Tool '{tn}' not found. Available: {', '.join(_VIZ_TOOLS)}."
                if isinstance(result, str):
                    trace.append({"type": "observation", "content": result})
                    msgs.append({"role": "user", "content": f"Observation: {result}"})
                else:
                    trace.append({"type": "image", "content": result})
                    msgs.append({"role": "user", "content": "Observation: Plot generated successfully."})
            else:
                trace.append({"type": "thought", "content": text})
                break
        return trace

    def _viz_model(messages, config):
        trace = _run_viz_agent(str(messages[-1].content))
        parts = []
        for step in trace:
            if step["type"] == "thought":
                parts.append(mo.md(f"> **Thought:** {step['content']}"))
            elif step["type"] == "action":
                parts.append(mo.md(f"> **Action:** `{step['tool']}` — input: `{step['input']}`"))
            elif step["type"] == "observation":
                parts.append(mo.md(f"> **Observation:** {step['content']}"))
            elif step["type"] == "image":
                parts.append(step["content"])
            elif step["type"] == "final":
                parts.append(mo.md(f"\n**Answer:** {step['content']}"))
            elif step["type"] == "error":
                parts.append(mo.md(f"> **Error:** {step['content']}"))
        return mo.vstack(parts) if parts else mo.md("*(No response — check the model config above.)*")

    viz_chat = mo.ui.chat(
        _viz_model,
        prompts=[
            "What columns are in the dataset?",
            "Show me a histogram of Age",
            "Show me a scatter plot of Age and Fare",
            "Plot a histogram of Survived, then a scatter plot of Pclass and Fare",
        ],
        show_configuration_controls=False,
    )
    viz_chat
    return (viz_chat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice what is happening here. The agent reads your plain-English question, decides
    which tool to call, calls it, and embeds the resulting plot directly in the reply.
    It can chain tool calls: ask it to first list the columns and then plot two of them,
    and you will see two separate Action steps before the Final Answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 3: Add Your Own Tool

    Now it is your turn. The cell below defines `my_tool` — a function that starts as
    a placeholder. Edit the body and the docstring to make it do something interesting.

    The docstring is the agent's only window into what your function does. Write it
    precisely: say what the function returns, what the argument means, and any
    constraints. Then use the chat below to ask a question that requires your tool.

    A few ideas to get you started. You could make `my_tool` return a fun fact about
    a Titanic column: "The oldest passenger was 80 years old." You could make it return
    a count of passengers matching a condition. You could make it do something completely
    unrelated to Titanic — return the number of vowels in a word, or convert Celsius to
    Fahrenheit. The agent will use whatever you build.

    /// tip | Try it yourself
    Edit `my_tool` below, then ask the chat agent to use it. Change the docstring too —
    the agent reads the docstring, not the function body. A well-written docstring is the
    difference between a tool that gets called correctly and one that gets ignored.
    ///
    """)
    return


@app.cell
def _(titanic_df):
    def my_tool(input_text: str) -> str:
        """
        Replace this docstring with a description of what your tool does.
        The agent reads this docstring to decide when and how to call your function.
        input_text: describe what this argument means, or write 'Takes no arguments'
        if you do not need one.
        """
        # Replace the line below with your own code.
        # You have access to titanic_df if you want to query the dataset.
        # Examples:
        #   return str(titanic_df["Age"].max())     # oldest passenger's age
        #   return str(titanic_df[input_text].mean()) # mean of any numeric column
        #   return f"{input_text!r} has {sum(c in 'aeiou' for c in input_text)} vowels"
        return f"my_tool was called with: {input_text!r}. Replace this with real code!"

    return (my_tool,)


@app.cell(hide_code=True)
def _(call_llm, list_columns, mo, my_tool, plot_histogram, plot_scatter, re):
    import inspect as _inspect3

    _ALL_TOOLS = {
        "list_columns": list_columns,
        "plot_histogram": plot_histogram,
        "plot_scatter": plot_scatter,
        "my_tool": my_tool,
    }

    _my_doc = (my_tool.__doc__ or "No description provided.").strip()

    _ALL_DESC = (
        "You are a helpful assistant with access to these tools:\n\n"
        "  1. list_columns() — returns all Titanic column names. Takes NO arguments.\n"
        "  2. plot_histogram(column) — draws a histogram of the given column. Returns an image.\n"
        "  3. plot_scatter(x_col, y_col) — draws a scatter plot of two columns. Returns an image.\n"
        f"  4. my_tool(input_text) — {_my_doc}\n\n"
        "Always respond in this exact format:\n"
        "Thought: <your reasoning>\n"
        "Action: <tool_name>\n"
        "Action Input: <argument(s), or 'none' if the tool takes no arguments.\n"
        "              For plot_scatter, separate columns with a comma: 'Age, Fare'>\n\n"
        "After receiving an Observation, continue with another Thought/Action or write:\n"
        "Final Answer: <your final answer to the user>"
    )

    def _run_all_agent(question: str, max_steps: int = 8) -> list:
        msgs = [{"role": "system", "content": _ALL_DESC}, {"role": "user", "content": question}]
        trace = []
        for _ in range(max_steps):
            try:
                text = call_llm(msgs).choices[0].message.content
            except Exception as e:
                trace.append({"type": "error", "content": str(e)})
                break
            msgs.append({"role": "assistant", "content": text})
            if "Final Answer:" in text:
                trace.append({"type": "final", "content": text[text.index("Final Answer:") + 13:].strip()})
                break
            if tm := re.search(r"Thought:\s*(.+?)(?:\nAction|$)", text, re.DOTALL):
                trace.append({"type": "thought", "content": tm.group(1).strip()})
            am = re.search(r"Action:\s*(.+)", text)
            im = re.search(r"Action Input:\s*(.+)", text)
            if am:
                tn = am.group(1).strip()
                ti = im.group(1).strip() if im else "none"
                trace.append({"type": "action", "tool": tn, "input": ti})
                if tn in _ALL_TOOLS:
                    try:
                        params = list(_inspect3.signature(_ALL_TOOLS[tn]).parameters)
                        if not params:
                            result = _ALL_TOOLS[tn]()
                        elif len(params) == 1:
                            result = _ALL_TOOLS[tn](ti)
                        else:
                            args = [a.strip() for a in ti.split(",", maxsplit=len(params) - 1)]
                            result = _ALL_TOOLS[tn](*args)
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = f"Tool '{tn}' not found. Available: {', '.join(_ALL_TOOLS)}."
                if isinstance(result, str):
                    trace.append({"type": "observation", "content": result})
                    msgs.append({"role": "user", "content": f"Observation: {result}"})
                else:
                    trace.append({"type": "image", "content": result})
                    msgs.append({"role": "user", "content": "Observation: Plot generated successfully."})
            else:
                trace.append({"type": "thought", "content": text})
                break
        return trace

    def _all_model(messages, config):
        trace = _run_all_agent(str(messages[-1].content))
        parts = []
        for step in trace:
            if step["type"] == "thought":
                parts.append(mo.md(f"> **Thought:** {step['content']}"))
            elif step["type"] == "action":
                parts.append(mo.md(f"> **Action:** `{step['tool']}` — input: `{step['input']}`"))
            elif step["type"] == "observation":
                parts.append(mo.md(f"> **Observation:** {step['content']}"))
            elif step["type"] == "image":
                parts.append(step["content"])
            elif step["type"] == "final":
                parts.append(mo.md(f"\n**Answer:** {step['content']}"))
            elif step["type"] == "error":
                parts.append(mo.md(f"> **Error:** {step['content']}"))
        return mo.vstack(parts) if parts else mo.md("*(No response — check the model config above.)*")

    my_chat = mo.ui.chat(
        _all_model,
        prompts=[
            "Use my_tool with any input you like",
            "Show me a histogram of Age, then use my_tool",
            "What columns are available?",
        ],
        show_configuration_controls=False,
    )
    my_chat
    return (my_chat,)
