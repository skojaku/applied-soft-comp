import marimo

__generated_with = "0.11.14-dev6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 🎯 Memory Challenge!

        You’ll be presented with a sequence of random integers, one by one. However, there’s a twist: **you can only see the last three integers, and you can only remember three numbers!** After the sequence ends, you'll face a randomly selected question about the sequence.

        ## 🧠 Your Challenge
        Develop a **smart strategy** to **compress key information** into just **three numbers** while the sequence unfolds. These values should help you answer different types of questions at the end.

        *Difficulty Levels*: Select your preferred difficulty level: **Easy, Medium, Hard**, etc. The difficulty affects the complexity of the questions you'll face.

        ## 🔄 How to Play
        1️⃣ **Plan Your Strategy** – Predict the types of questions you might get and decide what to track.
        2️⃣ **Track Three Numbers** 🔢 – As the numbers appear, update your notes wisely.
        3️⃣ **Answer a Question** – Use your stored values to answer a randomly selected question about the sequence.

        ⚡ Think fast, store wisely, and test your memory skills! 🚀
        """
    )
    return


@app.cell(hide_code=True)
def _(
    answer_box,
    create_bullet_list,
    create_highlighted_text,
    game,
    memo,
    mo,
    next_button,
    radiogroup,
    reset_button,
    submit_button,
    update_display,
):
    display = update_display(
        game.sequence,
        window_start=game.get_current_value(),
        window_size=game.stm_len,
        result=game.result,
    )


    possible_questions = create_bullet_list(
        "Question list",
        [q["question"] for q in game.questions],
        possible_values=game.integer_set,
        replacement=game.replacement,
    )

    question_box = mo.callout(game.get_message(), kind=game.get_message_type())

    # Example usage:
    # highlighted = create_highlighted_text("Your text here")
    submessage = create_highlighted_text(game.get_sub_message())

    mo.vstack(
        [
            radiogroup,
            possible_questions,
            mo.hstack([next_button, reset_button]),
            mo.hstack(display, justify="center"),
            mo.hstack(memo),
            question_box,
            submessage,
            mo.hstack([answer_box, submit_button]),
        ],
        align="center",
        justify="center",
    )
    return display, possible_questions, question_box, submessage


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 🔥 Design an LSTM for the memory game

        Now that you've tried the memory game, implement the solution using a simplified LSTM! Your code will mimic how you mentally tracked and processed the numbers.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## About LSTM""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <div align="center">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="-100 0 800 300">
            <!-- Styles -->
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
                </marker>
            </defs>


            <!-- Input Signal Line (without marker) -->
            <line x1="100" y1="280" x2="100" y2="220" stroke="#333" stroke-width="2"/>
            <text x="100" y="295" text-anchor="middle" font-size="14">input</text>



            <!-- Cell State to Output Gate -->
            <line x1="490" y1="80" x2="490" y2="120" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>

            <!-- Output Gate to Hidden State -->
            <line x1="470" y1="180" x2="470" y2="220" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>

            <!-- Output Gate to Output (offset to avoid overlap) -->
            <line x1="510" y1="180" x2="510" y2="215" stroke="#666" stroke-width="2"/>
            <line x1="510" y1="225" x2="510" y2="260" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>

            <text x="510" y="275" text-anchor="middle" font-size="12">output</text>

            <!-- Labels -->
            <text x="570" y="225" text-anchor="start" font-size="12">(last 3 numbers)</text>
            <text x="570" y="85" text-anchor="start" font-size="12">(stored statistics)</text>


            <!-- Cell State Line (top) -->
            <line x1="50" y1="80" x2="550" y2="80" stroke="#333" stroke-width="3"/>
            <text x="50" y="65" text-anchor="middle" font-size="14">cell_state</text>

            <!-- Hidden State Line (bottom) -->
            <line x1="50" y1="220" x2="550" y2="220" stroke="#333" stroke-width="3"/>
            <text x="50" y="235" text-anchor="middle" font-size="14">hidden_state</text>

            <!-- Operation Circles -->
            <circle cx="190" cy="80" r="12" fill="white" stroke="#333" stroke-width="2"/>
            <text x="190" y="85" text-anchor="middle" font-size="14">×</text>

            <circle cx="340" cy="80" r="12" fill="white" stroke="#333" stroke-width="2"/>
            <text x="340" y="85" text-anchor="middle" font-size="14">+</text>

                <!-- Forget Gate (pastel red) -->
            <rect x="150" y="120" width="80" height="60" fill="#FFB3B3" stroke="#333" stroke-width="2"/>
            <text x="190" y="155" text-anchor="middle" font-size="14">forget_gate</text>
            <line x1="190" y1="220" x2="190" y2="180" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
            <line x1="190" y1="120" x2="190" y2="92" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>

            <!-- Input Gate (pastel yellow) -->
            <rect x="300" y="120" width="80" height="60" fill="#FFF9C4" stroke="#333" stroke-width="2"/>
            <text x="340" y="155" text-anchor="middle" font-size="14">input_gate</text>
            <line x1="340" y1="220" x2="340" y2="180" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
            <line x1="340" y1="120" x2="340" y2="92" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>

            <!-- Output Gate (pastel blue) -->
            <rect x="450" y="120" width="80" height="60" fill="#BBDEFB" stroke="#333" stroke-width="2"/>
            <text x="490" y="155" text-anchor="middle" font-size="14">output_gate</text>
        </svg>
        </div>

        🔹 **LSTM (Long Short-Term Memory)** helps track and process information over time using key components:

        🛑 **Forget Gate (`forget_gate`)**

        - Decides what parts of the **cell_state** to **keep or erase**.
        - Outputs values between **0 (forget)** and **1 (keep)** for each stored item.

        ➕ **Input Gate (`input_gate`)**

        - Decides what **new information** to add to the **cell_state**.
        - Uses the current **input** and **hidden_state** to determine what updates to store.

        📦 **Cell State (`cell_state`)**

        - Stores long-term information.
        - Gets updated using the rule:

        📦 **Hidden State (`hidden_state`)**

        - Stores short-term information.
        - In our example, the hidden state holds the last three numbers.

        📤 **Output Gate (`output_gate`)**
        - Extracts useful information from the **cell_state**.
        - Updates the **hidden_state** to keep track of recent inputs.

        The cell states will be updated by the following way🥇

        `cell_state` := `cell_state` $\times$ `forget_factor` + `added_values`

        where

          - `forget_gate` decides what to erase: `cell_state × forget_factor`
          - `input_gate` decides what new values to add: `+ added_values`



        By controlling memory storage and updates, the LSTM **selectively** remembers important information while ignoring unnecessary details, just like when tracking numbers in a memory game 🎯
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## **Implementation Task 👨‍💻**

        LSTM maintains two key internal variables:

        - **`cell_state`**: A vector storing two of three memory notes you can keep.
        - **`hidden_state`**: A vector of length **two**, consisting of:
          - `hidden_state[0]`, `hidden_state[1]`: The **last two numbers** seen.
          - `hidden_state[2]`: A note you can keep.



        <div >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="-50 0 300 120" width="500">
            <!-- Definitions for reuse -->
            <defs>
                <rect id="square" width="25" height="25" rx="2"/>
            </defs>

            <!-- Cell State -->
            <g transform="translate(25, 25)">
                <!-- Squares -->
                <use href="#square" fill="#ff6b6b" stroke="#cc5555" stroke-width="1.5"/>
                <use href="#square" x="30" fill="#ff6b6b" stroke="#cc5555" stroke-width="1.5"/>

                <!-- Label -->
                <text x="-8" y="12" text-anchor="end" font-family="Arial" font-size="10">Cell state</text>

                <!-- User controlled annotation -->
                <text x="27" y="-6" text-anchor="middle" font-family="Arial" font-size="8" fill="#666">User controlled</text>
            </g>

            <!-- Hidden State -->
            <g transform="translate(25, 75)">
                <!-- Squares -->
                <use href="#square" fill="#a8c7f0"/>
                <use href="#square" x="30" fill="#a8c7f0"/>
                <use href="#square" x="60" fill="#4a90e3" stroke="#3a72b5" stroke-width="1.5"/>

                <!-- Labels -->
                <text x="-8" y="12" text-anchor="end" font-family="Arial" font-size="10">Hidden state</text>

                <!-- Last two numbers annotation -->
                <text x="27" y="-6" text-anchor="middle" font-family="Arial" font-size="8" fill="#666">Last two numbers</text>

                <!-- User controlled annotation -->
                <text x="72" y="35" text-anchor="middle" font-family="Arial" font-size="8" fill="#666">User controlled</text>
            </g>
        </svg>
        </div>

        **Example**: After processing the sequence `[5, 8, 12, 15]`:

        ```python
        # Managed by your implementation
        cell_state = [8, 12]  # Memory notes

        # Automatically updated
        hidden_state = [15, 12, # Last two numbers
                        3 # Managed by your implementation
                       ]
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### **Task**

        First, select the level (e.g., 'Easy', 'Medium') of questions for which your LSTM will answer.
        """
    )
    return


@app.cell(hide_code=True)
def _(radiogroup):
    radiogroup
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You will implement the following three functions:

        - **`forget_gate`**: Decide what to keep in your long-term memory.
        - **`input_gate`**: Choose what new information to store.
        - **`output_gate`**: Answer questions using your stored info.

        Your LSTM should generate answers and store them in the `output` array inside `output_gate`.

        - `output[i]` should contain the answer to the **\(i\)th** question.
        - For example, `output[0]` contains the first answer, `output[1]` contains the next, and so on.

        <div>
            <svg xmlns="http://www.w3.org/2000/svg" width="500" viewBox="-50 0 300 120">
            <!-- Definitions -->
            <defs>
                <rect id="cell" width="25" height="25" rx="2"/>
                <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
                        markerWidth="6" markerHeight="6" orient="auto">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#666"/>
                </marker>
            </defs>

            <!-- Output array -->
            <g transform="translate(25, 25)">
                <!-- Output cells -->
                <use href="#cell" fill="#ffa07a" stroke="#e67e5c" stroke-width="1.5"/>
                <use href="#cell" x="30" fill="#ffa07a" stroke="#e67e5c" stroke-width="1.5"/>
                <use href="#cell" x="60" fill="#ffa07a" stroke="#e67e5c" stroke-width="1.5"/>

                <!-- Output indices -->
                <text x="12" y="17" text-anchor="middle" font-family="Arial" font-size="10">[0]</text>
                <text x="42" y="17" text-anchor="middle" font-family="Arial" font-size="10">[1]</text>
                <text x="72" y="17" text-anchor="middle" font-family="Arial" font-size="10">[2]</text>

                <!-- Array label -->
                <text x="-8" y="12" text-anchor="end" font-family="Arial" font-size="10">output</text>
            </g>

            <!-- Questions -->
            <g transform="translate(25, 85)">
                <text x="12" y="0" text-anchor="middle" font-family="Arial" font-size="10">Q1</text>
                <text x="42" y="0" text-anchor="middle" font-family="Arial" font-size="10">Q2</text>
                <text x="72" y="0" text-anchor="middle" font-family="Arial" font-size="10">Q3</text>
            </g>

            <!-- Arrows connecting outputs to questions -->
            <g stroke="#666" stroke-width="1" marker-end="url(#arrow)">
                <line x1="37" y1="55" x2="37" y2="75"/>
                <line x1="67" y1="55" x2="67" y2="75"/>
                <line x1="97" y1="55" x2="97" y2="75"/>
            </g>

            <!-- User controlled annotation -->
            <text x="52" y="19" text-anchor="middle" font-family="Arial" font-size="8" fill="#666">User controlled</text>
        </svg>
        </div>

        Look for **🔥 MODIFY 🔥** markers in the code and implement your solution! 🚀
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Simplified LSTM: High-Level Implementation

        Before implementing each gate, let us first see how these gates are put together in an LSTM.
        """
    )
    return


@app.cell
def _(forget_gate, input_gate, output_gate):
    def longShortTermMemory(input, hidden_state, cell_state):
        """
        LSTM implementation using a stateless approach for processing sequences.
        Processes input through forget, input, and output gates.

        Args:
            input: Current input to the LSTM cell
                Single numeric value to process
            hidden_state: Hidden state from previous step
                Array [min_value, padding] tracking minimum value
            cell_state: Internal cell state from previous step
                Array tracking running [sum, product, min] values

        Returns:
            tuple:
                - hidden_state: Updated [min_value, padding] for next step
                - cell_state: Updated running [sum, product, min] values
                - output: Current [sum, product, min] outputs
        """
        # Forget gate: Determines what information to discard from the cell state
        # Returns values between 0 (forget) and 1 (keep) for each element
        forget_factor = forget_gate(input, hidden_state)
        assert all(
            0 <= x <= 1 for x in forget_factor
        ), "Forget gate outputs must be between 0 and 1"

        # Input gate: Decides what new information to store in the cell state
        # Processes current input and previous hidden state to generate new values
        added_values = input_gate(input, hidden_state)

        # Update cell state:
        # 1. Multiply old cell state by forget factor (forgetting irrelevant info)
        # 2. Add new values from input gate (adding new information)
        cell_state = cell_state * forget_factor + added_values

        # Output gate:
        # 1. Filters the cell state to determine what to output
        # 2. Updates the hidden state for the next time step
        output, hidden_state = output_gate(input, hidden_state, cell_state)

        return (
            hidden_state,  # [min_value, padding] for next step
            cell_state,  # Updated running [sum, product, min]
            output,  # Current [sum, product, min] outputs
        )
    return (longShortTermMemory,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Implementation of forget, input and output gates

        Let's proceed with the implementation of each gate.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        > To showcase how this works, the following starter code implements an LSTM for the questions at the entry-level 🧑‍🏫.
        > In this code, I used the memory as follows:
        >
        > 1. First cell state is used to answer the first question, i.e., the sum of all numbers
        > 2. Second cell state is used to answer the last question, i.e., the number of evens followed by the final odd
        > 3. I used the one hidden memory to answer the second question, i.e., the sum of evens

        You can remove and comment out the start code you want.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Hint for the 'Medium' mode": mo.md(
                "For LSTM implementation: Use 'hidden_state[2]' as a control variable that changes based on cell and hidden states. Its value (e.g., 0, positive, negative, or magnitude) determines how the LSTM processes and stores information."
            ),
            "Hint for the 'Expert' mode": mo.md(
                "Note that the median is the fourth smallest/largest number in the sequence. You'll need to utilize hidden_state[2] as a control variable in your LSTM logic."
            ),
        }
    )
    return


@app.cell
def _(np):
    def forget_gate(input, hidden_state):
        """
        Returns forgetting factors (0-1) for each cell state value.
        0 = forget, 1 = keep, values between = partial update
        Args:
            input: Current input value
            hidden_state: Previous outputs array [min_value, padding]
        """
        # 🔥 MODIFY: Set forgetting factors 🔥
        forgetting_factor = [
            1.0,  # no forgetting takes place since we want to compute the sum of all numbers.
            0.0 if input %2 != 0 else 1.0,  # we need to forget the count when input is odd
        ]

        return np.array(forgetting_factor)


    def input_gate(input, hidden_state):
        """
        Returns new values to add to each cell state position.
        Args:
            input: Current input value
            hidden_state: Previous outputs array [min_value, padding]
        """
        # 🔥 MODIFY: Set values to add 🔥
        addition_cell_state = [
            input,  # We add the input since we are interested in the sum of all numbers
            0.0 if input %2 !=0 else 1.0,  # We want to add 1 when the input is even
        ]

        return np.array(addition_cell_state)


    def output_gate(input, hidden_state, cell_state):
        """
        Returns (outputs, hidden_state).
        Args:
            input: Current input value
            hidden_state: Previous outputs array [min_value, padding]
            cell_state: Current cell state values
        """


        # 🔥 MODIFY: Set your hidden state 🔥
        your_hidden_state = hidden_state[2]
        your_hidden_state+= input if input %2 == 0 else 0.0 # Here we use the hidden state as a counter, where we add the input to the counter if the input is even, since we are interested in the sum of evens.

        # 🔥 MODIFY: Set outputs 🔥
        output = [cell_state[0], your_hidden_state, cell_state[1]]

        hidden_state = [
            hidden_state[1],  # Leave this as is
            input,  # Leave this as is
            your_hidden_state,  # Leave this as is
        ]
        return np.array(output), np.array(hidden_state)
    return forget_gate, input_gate, output_gate


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Evaluation

        Let's test the implemented LSTM
        """
    )
    return


@app.cell(hide_code=True)
def _(game, longShortTermMemory, np, radiogroup):
    num_tests = 100
    hidden_state_size = 3
    cell_state_size = 2


    game.set_level(radiogroup.value)

    def run_lstm(sequence, hidden_state_size, cell_state_size):
        """
        Run LSTM over a sequence of inputs.

        Args:
            sequence: Input sequence to process
            hidden_state_size: Size of hidden state array [min_value, padding]
            cell_state_size: Size of cell state array [sum, product, min]

        Returns:
            output: Final [sum, product, min] outputs after processing sequence
        """
        hidden_state = np.zeros(
            hidden_state_size
        )  # Initialize [min_value, padding]
        cell_state = np.zeros(cell_state_size)  # Initialize [sum, product, min]

        for s in sequence:
            hidden_state, cell_state, output = longShortTermMemory(
                s, hidden_state, cell_state
            )
        return output


    # Run tests and compute accuracy
    n_questions = len(game.questions)
    n_correct = np.zeros(n_questions, dtype=float)
    for i in range(num_tests):
        game.reset()
        output = run_lstm(game.sequence, hidden_state_size, cell_state_size)
        answers = np.array([q["func"]() for q in game.questions])

        output = np.array(output)[: len(answers)]
        # Compute the accuracy for each run
        n_correct += np.isclose(answers, output, atol=1e-1)

    n_correct /= num_tests
    final_score = np.min(n_correct)
    return (
        answers,
        cell_state_size,
        final_score,
        hidden_state_size,
        i,
        n_correct,
        n_questions,
        num_tests,
        output,
        run_lstm,
    )


@app.cell(hide_code=True)
def _(final_score, game, mo, n_correct, n_questions, radiogroup):
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, NumeralTickFormatter

    # Convert range to list for x values
    x_values = list(range(len(n_correct)))

    # Pastel colors
    pastel_colors = [
        "#FFB3B3",  # pastel red
        "#BBDEFB",  # pastel blue
        "#FFF9C4",
    ]  # pastel yellow

    pastel_colors = [p for i, p in enumerate(pastel_colors) if i < n_questions]

    # Calculate total score (second smallest accuracy)
    game_level = ["Entry-Level", "Easy", "Medium", "Expert", "Master"][
        radiogroup.value - 1
    ]

    # Create the data source
    source = ColumnDataSource(
        {
            "x": x_values,
            "y": n_correct,
            "question": [q["question"] for q in game.questions],
            "accuracy_pct": [f"{acc:.1%}" for acc in n_correct],
            "colors": [
                pastel_colors[i % len(pastel_colors)]
                for i in range(len(n_correct))
            ],
        }
    )

    # Create figure with numerical x-range
    p = figure(
        height=400,
        title="LSTM Model Accuracy by Question",
        tools="hover",
        tooltips=[("Question", "@question"), ("Accuracy", "@accuracy_pct")],
    )

    # Add bars with color from data source
    p.vbar(
        x="x",
        top="y",
        width=0.5,
        source=source,
        fill_color="colors",
        line_color="colors",
    )

    # Customize
    p.xaxis.axis_label = "Question Number"
    p.yaxis.axis_label = "Accuracy"
    p.yaxis.formatter = NumeralTickFormatter(format="0%")
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = 1.0

    # Set x-axis ticks
    p.xaxis.ticker = x_values
    p.xaxis.major_label_overrides = {i: f"Q{i+1}" for i in x_values}

    # Style the plot
    p.title.text_font_size = "14pt"
    p.background_fill_color = "white"
    p.border_fill_color = "white"
    p.grid.grid_line_color = "#f5f5f5"

    # Create the final stacked output
    mo.vstack(
        [
            mo.md(f"""## LSTM Performance Results
        - **Level**: {game_level}
        - **Total Score**: {final_score:.1%}
        """),
            p,
        ],
        gap=2,
    )
    return (
        ColumnDataSource,
        NumeralTickFormatter,
        figure,
        game_level,
        p,
        pastel_colors,
        show,
        source,
        x_values,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Backend""")
    return


@app.cell(hide_code=True)
def _(game, is_numbers, mo):
    radiogroup = mo.ui.radio(
        options={
            "Entry-Level": 1,
            "Easy": 2,
            "Medium": 3,
            "Expert": 4,
            # "Master": 5,
        },
        value="Easy",
        label="Level: ",
        inline=True,
        on_change=lambda value: game.set_level(value),
    )

    next_button = mo.ui.button(
        value=0,
        on_click=lambda value: game.update_current_index(),
        label="Next Number",
    )

    reset_button = mo.ui.button(
        value=0,
        on_click=lambda value: game.reset(),
        label="Reset",
    )

    answer_box = mo.ui.text(placeholder="Answer", label="", value="")

    submit_button = mo.ui.button(
        value=0,
        on_click=lambda value: game.eval(answer_box.value),
        label="Submit",
    )

    memo = [
        mo.ui.text(
            placeholder="Memo",
            label="",
            value="",
            on_change=lambda value: value if is_numbers(value) else "",
        )
        for _ in range(game.stm_len)
    ]
    return (
        answer_box,
        memo,
        next_button,
        radiogroup,
        reset_button,
        submit_button,
    )


@app.cell(hide_code=True)
def _(MemoryGame):
    game = MemoryGame()
    game.set_level(1)
    return (game,)


@app.cell(hide_code=True)
def _(np, selq):
    class MemoryGame:
        def __init__(
            self, seq_len=10, stm_len=3, integer_set=list(range(10)), level=1
        ):
            self.seq_len = seq_len
            self.stm_len = stm_len
            self.integer_set = list(set(integer_set))
            self.current_index = None
            self.sequence = None
            self.stm = None
            self.question_index = None
            self.result = "none"
            self.level = 1
            self.replacement = True

            self.questions = []
            self.question_pool = [
                # Level 1 -------------------------------------------------------
                {
                    "question": "What was the sum of all numbers?",
                    "func": self.get_sum,
                    "level": 1,
                },
                {
                    "question": "What was the sum of even numbers ",
                    "func": lambda: np.sum(
                        [s for s in self.sequence if s % 2 == 0]
                    ),
                    "level": 1,
                },
                {
                    "question": "How many evens followed the final odd? (Total length if no odds)",
                    "func": lambda: len(self.sequence)
                    - np.where(np.array(self.sequence) % 2 == 1)[0][-1]
                    - 1
                    if any(np.array(self.sequence) % 2 == 1)
                    else len(self.sequence),
                    "level": 1,
                },
                # Level 2 -------------------------------------------------------
                {
                    "question": "How many odd numbers in a row at their last appearance?",
                    "func": self.get_len_last_odd_in_row,
                    "level": 2,
                },
                {
                    "question": "What was the most frequent number in the sequence? If ties, answer the smallest one.",
                    "func": self.get_mode,
                    "level": 2,
                },
                # Level 2 -------------------------------------------------------
                {
                    "question": "What was the smallest number in the first four numbers?",
                    "func": lambda: np.min(self.sequence[:4]),
                    "level": 3,
                },
                {
                    "question": "What was the largest number in the last four numbers?",
                    "func": lambda: np.max(self.sequence[-4:]),
                    "level": 3,
                },
                # Level 4 -------------------------------------------------------
                {
                    "question": "What was the median?",
                    "func": self.get_median,
                    "level": 4,
                },
            ]
            self.hints = [
                "",
                "To solve this problem, consider how you would track and count the frequency of each number given the numbers and sequence length.",
                "For LSTM implementation: Use 'hidden_state[2]' as a control variable to manage LSTM behavior. When hidden_state[2] = 0, the LSTM will store values in memory. When hidden_state[2] > 0, the LSTM will perform calculations to summarize stored values. When hidden_state[2] < 0, the LSTM will modify its compression operation. You can define how 'hidden_state[2]' transitions between states based on the current cell and hidden states in the output gate.",
                "For LSTM implementation: Note that the median is the fourth smallest/largest number in the sequence. You'll need to utilize hidden_state[2] as a control variable in your LSTM logic.",
            ]

            self.reset()

        def set_level(self, level):
            if level == 1:  # Easy
                self.reset(
                    seq_len=7,
                    stm_len=3,
                    integer_set={1, 2, 4, 8},
                    level=level,
                    replacement=True,
                )
            if level == 2:  # Medium (modified)
                self.reset(
                    seq_len=7,
                    stm_len=3,
                    integer_set={1, 2, 3},
                    level=level,
                    replacement=True,
                )
            if level == 3:  # Medium (modified)
                self.reset(
                    seq_len=7,
                    stm_len=3,
                    integer_set={1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                    level=level,
                    replacement=False,
                )
            if level == 4:  # Master
                self.reset(
                    seq_len=7,
                    stm_len=3,
                    integer_set={1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                    level=level,
                    replacement=False,
                )

        def update_stm(self, number):
            self.stm.append(number)
            if len(self.stm) > self.stm_len:
                self.stm.pop(0)

        def get_current_value(self):
            return self.current_index

        def reset(
            self,
            seq_len=None,
            stm_len=None,
            integer_set=None,
            level=None,
            replacement=None,
        ):
            self.seq_len = seq_len if seq_len is not None else self.seq_len
            self.stm_len = stm_len if stm_len is not None else self.stm_len
            self.replacement = (
                replacement if replacement is not None else self.replacement
            )
            self.integer_set = (
                list(set(integer_set))
                if integer_set is not None
                else self.integer_set
            )
            self.level = level if level is not None else self.level

            self.sequence = np.random.choice(
                self.integer_set, size=self.seq_len, replace=self.replacement
            )
            self.stm = []
            self.current_index = 0

            self.questions = [
                q for q in self.question_pool if q["level"] == self.level
            ]
            self.question_index = np.random.randint(0, len(self.questions))
            self.result = "none"

        def update_current_index(self):
            if self.current_index < self.seq_len - 1:
                self.current_index += 1

        def get_sub_message(self):
            question = self.questions[self.question_index]["question"]
            answer = self.questions[self.question_index]["func"]()
            answer = float(answer)

            if self.result == "correct":
                return f"""
                Correct! 🥳
                """

            elif self.result == "incorrect":
                return f"""
                Incorrect 🤔. The answer was {answer:.1f}
                """

            return ""

        def get_message(self):
            progress = self.current_index + 1
            numbers_left = self.seq_len - progress - self.stm_len + 1
            message = ""

            question = self.questions[self.question_index]["question"]
            answer = self.questions[self.question_index]["func"]()
            answer = float(answer)

            if self.current_index < self.seq_len - self.stm_len:
                if self.current_index == 0:
                    message = "Remember the numbers as they appear! "
                else:
                    message = f"{numbers_left} more number{'s' if numbers_left > 1 else ''} to go "
            else:
                message = self.questions[self.question_index]["question"]
            return message

        def get_message_type(self):
            if self.current_index < self.seq_len - self.stm_len:
                return "neutral"
            elif self.result == "correct":
                return "success"
            elif self.result == "incorrect":
                return "danger"
            else:
                return "info"

        def eval(self, answer):
            try:
                answer = float(answer)
            except ValueError:
                return

            correct_answer = self.questions[self.question_index]["func"]()
            if np.abs(answer - correct_answer) < 0.1:
                self.result = "correct"
            else:
                self.result = "incorrect"

        def get_smallest(self):
            return np.min(self.sequence)

        def get_largest(self):
            return np.max(self.sequence)

        def get_average(self):
            return np.mean(self.sequence)

        def get_product(self):
            return np.prod(self.sequence)

        def get_sum(self):
            return np.sum(self.sequence)

        def get_sum_squared(self):
            return np.sum(self.sequence**2)

        def get_second_smallest(self):
            return np.sort(self.sequence)[1]

        def get_median(self):
            return np.median(self.sequence)

        def get_num_numbers_after_last_odd(self):
            cnt = 0
            for i in range(len(self.sequence)):
                if self.sequence[-(i + 1)] % 2 == 0:
                    cnt += 1
                break
            return cnt

        def get_len_last_odd_in_row(self):
            cnt = 0
            odd_count = 0
            for i in range(len(self.sequence)):
                if self.sequence[i] % 2 == 0:
                    odd_count = cnt
                    cnt = 0
                else:
                    cnt += 1
            return odd_count

        def get_last_four_pos_number(self):
            return selq.sequence[-4]

        def get_mode(self):
            labs, freq = np.unique(self.sequence, return_counts=True)
            indices = np.where(np.max(freq) == freq)[0]
            return labs[indices[0]]
    return (MemoryGame,)


@app.cell(hide_code=True)
def _(mo, np):
    def create_box(
        num="?",
        boxsize=60,
        color="#2d2d2d",
        is_masked=False,
        is_last_unmasked=False,
    ):
        """
        Creates a box that can display a number or mask
        Args:
            num: Number or symbol to display
            boxsize: Size of the box
            color: Box color
            is_masked: Whether to show mask instead of number
        """
        box_size = boxsize
        display_value = "?" if is_masked else str(num)

        # Color for masked vs unmasked boxes
        if is_masked:
            gradient = (
                "rgba(209, 213, 219, 0.7), #D1D5DB"  # Gray gradient for masked
            )
        else:
            gradient = "rgba(135, 206, 250, 0.5), #87CEFA"  # More saturated pastel blue gradient

        if is_last_unmasked:
            gradient = "rgba(255, 250, 160, 0.7), #FFF5A0"  # Natural yellow pastel for emphasis

        styles = f"""
            display: flex;
            align-items: center;
            justify-content: center;
            width: {box_size}px;
            height: {box_size}px;
            background: linear-gradient(135deg, {gradient});
            border-radius: 12px;
            font-family: 'Playfair Display', 'Merriweather', 'Crimson Text', Georgia, serif;
            font-size: {max(16, min(box_size//2, 24))}px;
            font-weight: 500;
            color: {color};
            border: 1px solid rgba(255, 255, 255, 0.1);
            cursor: default;
            flex-shrink: 0;
            letter-spacing: 0.5px;
        """

        return mo.Html(f"""
            <div style="{styles}">{display_value}</div>
        """)


    def display_sequence(numbers, masks=None, boxsize=60):
        """
        Display a sequence of numbers with optional masks
        Args:
            numbers: List of integers to display
            masks: List of booleans indicating which positions to mask (True = masked)
            boxsize: Size of each box
        """
        if masks is None:
            masks = [False] * len(numbers)

        container_style = """
            gap: 10px;
            padding: 10px;
            background: #f8fafc;
            border-radius: 16px;
        """

        boxes = [
            create_box(num=num, boxsize=boxsize, is_masked=mask)
            for num, mask in zip(numbers, masks)
        ]
        return boxes


    def update_display(sequence, window_start, window_size, result):
        window_start = np.minimum(window_start, len(sequence) - window_size)
        # Create masks based on window position
        if result == "none":
            masks = [True] * len(sequence)
            for i in range(
                window_start, min(window_start + window_size, len(sequence))
            ):
                masks[i] = False
        else:
            masks = [False] * len(sequence)

        # Create boxes with updated masks
        boxes = [
            create_box(
                num=num,
                boxsize=60,
                is_masked=mask,
                is_last_unmasked=True
                if i == window_start + window_size - 1
                else False,
            )
            for i, (num, mask) in enumerate(zip(sequence, masks))
        ]
        return boxes


    def create_highlighted_text(text: str) -> mo.Html:
        """
        Creates text with a yellow highlight underline effect.

        Args:
            text: The text to be highlighted

        Returns:
            mo.Html: Marimo HTML element with highlighted text
        """
        html = f"""
            <div style="
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                display: inline-block;
                position: relative;
            ">
                <span style="
                    position: relative;
                    z-index: 1;
                    font-size: 18px;
                    color: #333;
                ">{text}</span>
                <span style="
                    position: absolute;
                    bottom: 2px;
                    left: 0;
                    width: 100%;
                    height: 8px;
                    background-color: rgba(255, 235, 59, 0.5);
                    z-index: 0;
                "></span>
            </div>
        """
        return mo.Html(html)


    def create_bullet_list(
        title: str,
        items: list,
        possible_values: list,
        replacement: bool,
        title_color: str = "#333333",
        bullet_color: str = "#555555",
        title_size: str = "24px",
    ) -> mo.md:
        """
        Creates a styled centered bullet list with title using HTML in Marimo markdown
        Args:
            title: Title text to display above the list
            items: List of strings to display as bullet points
            possible_values: List of possible values to display
            replacement: Boolean indicating if duplicates are allowed
            title_color: Color for the title (default: dark gray)
            bullet_color: Color for the bullet points (default: medium gray)
            title_size: Font size for the title (default: 24px)
        Returns:
            mo.
            down: Marimo markdown element with styled title and bullet list
        """
        bullet_points = "\n".join([f"<li>{item}</li>" for item in items])
        possible_values = ", ".join(
            [
                f"<span style='background: #f5f5f5; padding: 2px 8px; border-radius: 4px; margin: 0 2px'>{item}</span>"
                for item in possible_values
            ]
        )

        duplication_text = (
            "Duplicates are allowed"
            if replacement
            else "Duplicates are **NOT** allowed"
        )

        html_text = f"""
    <div>
        <div style="width: 100%; text-align: center;">
            <h2 style="color: {title_color}; font-size: {title_size}; margin-bottom: 20px; display: inline-block;">
                {title}
            </h2>
        </div>
        <div style="display: flex; justify-content: center;">
            <ul style="text-align: left; color: {bullet_color};
                       line-height: 1.6; margin: 0 auto; padding-left: 20px;">
                {bullet_points}
            </ul>
        </div>
        <div style="width: 100%; text-align: center; margin-top: 20px;">
            <div style="display: inline-block; text-align: left; color: {bullet_color};">
                <span style="font-weight: 500;">Possible values:</span> {possible_values} ({duplication_text})
            </div>
        </div>
        <hr style="margin-top: 20px; border: 0; height: 1px; background: #cccccc;">
    </div>
    """
        return mo.md(html_text)


    def create_horizontal_line(
        color: str = "#cccccc",
        height: str = "1px",
        margin: str = "20px 0",
        width: str = "100%",
    ) -> mo.md:
        """
        Creates a horizontal line with customizable styling
        Args:
            color: Color of the line (default: light gray)
            height: Height/thickness of the line (default: 1px)
            margin: Margin around the line (default: 20px top and bottom)
            width: Width of the line (default: 100%)
        Returns:
            mo.Markdown: Marimo markdown element with horizontal line
        """
        html_text = f"""
    <hr style="border: 0;
               height: {height};
               background: {color};
               margin: {margin};
               width: {width};">
    """
        return mo.md(html_text)


    def is_numbers(text):
        # Handle empty strings or None
        if not text:
            return False

        # Check if string contains only digits
        return text.isdigit()
    return (
        create_box,
        create_bullet_list,
        create_highlighted_text,
        create_horizontal_line,
        display_sequence,
        is_numbers,
        update_display,
    )


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import random
    return mo, np, random


@app.cell(hide_code=True)
def _():
    def configure_game(level, game):
        if level == 1:  # Easy
            game.reset(seq_len=7, stm_len=3, integer_set={1, 2, 3}, level=level)
        if level == 2:  # Medium
            game.reset(seq_len=7, stm_len=3, integer_set={1, 2, 3}, level=level)
        if level == 3:  # Medium
            game.reset(seq_len=7, stm_len=2, integer_set={1, 2, 3}, level=level)
        if level == 4:  # Hard
            game.reset(
                seq_len=9, stm_len=3, integer_set={1, 2, 3, 4, 5, 6}, level=level
            )
    return (configure_game,)


if __name__ == "__main__":
    app.run()
