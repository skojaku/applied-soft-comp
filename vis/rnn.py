import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 🧠 Learn RNNs Through Physics!

        <div style="display: flex; justify-content: center;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 350" width ="700">
            <!-- Definitions -->
            <defs>
                <!-- Arrow marker -->
                <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                        refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
                </marker>
            </defs>
            <!-- Background -->
            <rect width="700" height="300" fill="#fff"/>
        
            <!-- Left side: Physical System -->
            <g transform="translate(80,80)">
                <!-- Wall -->
                <rect x="0" y="90" width="20" height="100" fill="#666"/>
            
                <!-- Spring (connected to mass) -->
                <path d="M20,130 C30,130 30,120 40,120 S50,130 60,130 S70,120 80,120 S90,130 100,130 S110,120 120,120 S130,130 140,130" 
                      stroke="#666" fill="none" stroke-width="2"/>
            
                <!-- Shortened and raised damper -->
                <path d="M20,155 L50,155" stroke="#666" stroke-width="2"/>
                <!-- Damper piston -->
                <rect x="50" y="147" width="25" height="16" fill="#666"/>
                <!-- Damper cylinder -->
                <path d="M75,148 L110,148 L110,162 L75,162" stroke="#666" fill="none" stroke-width="2"/>
                <!-- Straight connection to mass -->
                <line x1="110" y1="155" x2="140" y2="155" stroke="#666" stroke-width="2"/>
            
                <!-- Mass (increased size) -->
                <rect x="140" y="110" width="70" height="70" fill="#4664c5" stroke="none"/>
                <text x="175" y="150" text-anchor="middle" fill="white" font-family="Arial" font-size="16">m</text>
            
                <!-- Force arrow (attached to mass) -->
                <line x1="210" y1="140" x2="240" y2="140" stroke="#d64242" 
                      marker-end="url(#arrowhead)" stroke-width="2"/>
                <text x="225" y="125" fill="#d64242" font-family="Arial">F</text>
            
                <!-- Label -->
                <text x="80" y="50" text-anchor="middle" font-family="Arial" 
                      font-size="20">Physical System</text>
            </g>
            <!-- Right side: RNN -->
            <g transform="translate(400,80)">
                <!-- Hidden State -->
                <rect x="100" y="100" width="120" height="60" rx="10" 
                      fill="#4664c5" stroke="none"/>
                <text x="160" y="130" text-anchor="middle" fill="white" font-family="Arial">Hidden State</text>
                <text x="160" y="150" text-anchor="middle" fill="white" font-family="Arial" font-size="12">h₁(t), h₂(t)</text>
            
                <!-- Input (bottom to top) -->
                <rect x="130" y="200" width="60" height="30" rx="5" 
                      fill="#8b8b8b" stroke="none"/>
                <text x="160" y="220" text-anchor="middle" fill="white" font-family="Arial">u(t)</text>
            
                <!-- Output (bottom to top) -->
                <rect x="130" y="30" width="60" height="30" rx="5" 
                      fill="#e85757" stroke="none"/>
                <text x="160" y="50" text-anchor="middle" fill="white" font-family="Arial">y(t)</text>
            
                <!-- Connections -->
                <line x1="160" y1="200" x2="160" y2="160" stroke="#333" 
                      marker-end="url(#arrowhead)" fill="none"/>
                <line x1="160" y1="100" x2="160" y2="60" stroke="#333" 
                      marker-end="url(#arrowhead)" fill="none"/>
            
                <!-- Modified self-loop direction -->
                <path d="M100,130 C50,130 50,100 100,100" 
                      stroke="#333" marker-end="url(#arrowhead)" fill="none"/>
            
                <!-- Label -->
                <text x="160" y="0" text-anchor="middle" font-family="Arial" 
                      font-size="20">RNN Structure</text>
            </g>
            <!-- Center arrow and label - vertically aligned with other figures -->
            <g transform="translate(300,130)">
                <line x1="40" y1="80" x2="140" y2="80" stroke="#333" 
                      marker-end="url(#arrowhead)" stroke-width="2"/>
                <text x="80" y="70" text-anchor="middle" font-family="Arial" 
                      font-size="16">Mapping</text>
            </g>
        </svg>
        </div>

        We’ll design a simple recurrent neural network (RNN) to model the motion of an object attached to a spring and damper. When displaced and released, the object oscillates with decaying amplitude.

        The system equation for the physical modition is as follow (with Euler integration with time step $dt$ ):

        $$
        \\begin{align*}
        v(t + dt) &= v(t) + \\left[-k \\cdot x(t) - c \\cdot v(t) + u(t) \\right] dt \\\\
        x(t + dt) &= x(t) + v(t) dt
        \\end{align*}
        $$

        where $u(t)$ is the external force, $k$ is the spring coefficient, and $c$ is the damping factor.

        Now, let us consider an RNN that follows the recurrent equations:

        $$
        \\begin{align*}
        h(t+1) = f\\left(h(t), u(t)\\right), \\quad y(t+1) = g(h(t+1))
        \\end{align*}
        $$

        where $h(t)$ is a hidden state with 2 hidden variables:

        $$
        h(t)= \\begin{bmatrix}
        h_1(t) \\\\
        h_2(t)
        \\end{bmatrix}
        $$

        Function $f$ creates the recurrency, which computes the next hidden state based on the current hidden state and the input. Here, let us assume:

        $$
        \\begin{align*}
        f \\left( h(t), u(t) \\right)
        :=
        \\begin{bmatrix}
        w_{11} & w_{12} & w_{13} \\\\
        w_{21} & w_{22} & w_{23}
        \\end{bmatrix}
        \\begin{bmatrix}
        h_1(t) \\\\
        h_2(t) \\\\
        u(t)
        \\end{bmatrix}
        \\end{align*}
        $$

        where $w_{ij}$ are the parameters of the RNN.

        Function $g$ acts as a readout, producing the output from the hidden states. We will assume:

        $$
        g(h(t)) =
        \\begin{bmatrix}
        r_{11} & r_{12} \\
        \\end{bmatrix}
        h(t)
        $$

        ### 🔥Task🔥

        Design $w_{ij}$ and $r_{ij}$ to produce the damped oscillatory motion of a mass-spring-damper system. The goal is to generate a trajectory that mimics the system’s natural behavior.
        """
    )
    return


@app.cell
def _():
    import numpy as np


    # Define RNN dynamics
    def f_recurrent(h, u, params):
        k = params["spring_coef"]
        c = params["damping_factor"]
        dt = params["dt"]

        # --------------
        # #TODO: Design the parameters to create sinusoidal wave
        W = [[0, 0, 0], [0, 0, 0]]  #
        # ---------------

        x = np.concatenate([h, [u]])
        new_h = np.array(W) @ x.reshape((-1, 1))
        new_h = new_h.reshape(-1)

        return new_h


    def g_recurrent(h):
        # --------------
        # #TODO: Design the parameters to create sinusoidal wave
        R = [[0, 0]]  #
        # ---------------
        y = np.array(R) @ h

        return y


    # Cell for generating time series data
    def generate_time_series(ut, h0, params):
        # Initialize empty lists to store outputs and hidden states
        outputs, hidden_states = [], []
        # Set initial hidden state
        h = h0  # initial state

        # Iterate through each timestep
        for t in range(len(ut)):
            # Update hidden state using recurrent function f
            # Takes current hidden state h, input ut[t], and system parameters
            h = f_recurrent(h, ut[t], params)

            # Generate output for current timestep using function g
            # Maps hidden state to output space
            output = g_recurrent(h)

            # Store hidden state and output for this timestep
            hidden_states.append(h)
            outputs.append(output)

        # Convert lists to numpy arrays
        # Stack hidden states vertically - shape will be (timesteps, hidden_dim)
        hidden_states = np.vstack(hidden_states)
        # Stack outputs vertically - shape will be (timesteps, output_dim)
        outputs = np.vstack(outputs)

        return outputs, hidden_states


    # Initial hidden state
    h0 = np.array([1, 0])
    return f_recurrent, g_recurrent, generate_time_series, h0, np


@app.cell(hide_code=True)
def _(
    damping_slider,
    generate_time_series,
    h0,
    np,
    radiogroup,
    spring_slider,
    visualize_rnn,
):
    params = {
        "spring_coef": spring_slider.value,
        "damping_factor": damping_slider.value,
        "dt": 0.01,
    }


    # Input
    ut = np.zeros(10000)

    if radiogroup.value == 2:
        ut += 1
    elif radiogroup.value == 3:
        a = 0.01
        ut = np.sin(np.arange(len(ut)) * a)

    visualize_rnn(ut, h0, params, generate_time_series)
    return a, params, ut


@app.cell(hide_code=True)
def _(mo):
    # Create interactive elements
    spring_slider = mo.ui.slider(
        start=-1.5, stop=1.5, step=0.01, value=1.0, label="Spring coefficient, k"
    )

    # Create interactive elements
    damping_slider = mo.ui.slider(
        start=-1.5, stop=1.5, step=0.01, value=1.0, label="Damping factor, c"
    )

    radiogroup = mo.ui.radio(
        options={"u(t)=0": 1, "u(t)=sin(at)": 3},
        value="u(t)=0",
        label="Pick input u(t)",
    )

    mo.vstack([mo.hstack([spring_slider, damping_slider]), radiogroup])
    return damping_slider, radiogroup, spring_slider


@app.cell(hide_code=True)
def _(np):
    import altair as alt
    import pandas as pd
    import marimo as mo


    def visualize_rnn(ut, h0, params, generate_time_series):
        """
        Visualize RNN time series data using Vega/Altair.
        """
        # Set data transformer to handle larger datasets
        alt.data_transformers.disable_max_rows()

        # Generate time series data
        outputs, hidden_states = generate_time_series(ut, h0, params)

        # Downsample data if necessary
        max_points = 300
        if len(ut) > max_points:
            step = len(ut) // max_points
            time_points = np.arange(0, len(ut), step)
            ut = ut[::step]
            outputs = outputs[::step]
            hidden_states = hidden_states[::step]
        else:
            time_points = np.arange(len(ut))

        # Create DataFrames
        df_io = pd.DataFrame(
            {"Time Steps": time_points, "Input": ut, "Output": outputs.reshape(-1)}
        ).melt(id_vars=["Time Steps"], var_name="Series", value_name="Value")

        df_hidden = pd.DataFrame(
            {
                "Time Steps": time_points,
                "h₁": hidden_states[:, 0],
                "h₂": hidden_states[:, 1],
            }
        ).melt(id_vars=["Time Steps"], var_name="Hidden State", value_name="Value")

        # Common chart properties
        width = 600
        height = 150

        # Create input/output plot
        io_plot = (
            alt.Chart(df_io)
            .mark_line()
            .encode(
                x=alt.X("Time Steps:Q", title="Time Steps"),
                y=alt.Y("Value:Q", title="Value"),
                color=alt.Color(
                    "Series:N",
                    scale=alt.Scale(
                        domain=["Input", "Output"], range=["#1D84B5", "#E6425E"]
                    ),
                    legend=alt.Legend(
                        orient="bottom",
                        title="Series",
                        labelFontSize=12,
                        titleFontSize=14,
                        offset=10,
                    ),
                ),
                tooltip=["Time Steps:Q", "Value:Q", "Series:N"],
            )
            .properties(
                width=width, height=height, title="Input/Output Time Series"
            )
        )

        # Create hidden states plot
        hidden_plot = (
            alt.Chart(df_hidden)
            .mark_line()
            .encode(
                x=alt.X("Time Steps:Q", title="Time Steps"),
                y=alt.Y("Value:Q", title="Value"),
                color=alt.Color(
                    "Hidden State:N",
                    scale=alt.Scale(
                        domain=["h₁", "h₂"], range=["#FFA500", "#FF7F50"]
                    ),
                    legend=alt.Legend(
                        orient="bottom",
                        title="Hidden State",
                        labelFontSize=12,
                        titleFontSize=14,
                        offset=10,
                    ),
                ),
                tooltip=["Time Steps:Q", "Value:Q", "Hidden State:N"],
            )
            .properties(width=width, height=height, title="Hidden States")
        )

        # Combine the plots vertically
        combined_plot = (
            alt.vconcat(io_plot, hidden_plot)
            .configure_axis(grid=True)
            .configure_view(strokeWidth=0)
            .configure_legend(
                symbolSize=100,
                labelLimit=200,
                padding=10,
                cornerRadius=4,
                strokeColor="gray",
                fillColor="white",
                offset=5,
            )
            .properties(
                padding={"left": 50, "top": 30, "right": 100, "bottom": 40}
            )
            .interactive()
        )

        return combined_plot
    return alt, mo, pd, visualize_rnn


if __name__ == "__main__":
    app.run()
