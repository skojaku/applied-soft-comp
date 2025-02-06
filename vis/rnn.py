import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _(np):
    # Define RNN dynamics
    def rnn_step(h, w, u, use_nonlin=False):
        position = h[0]
        velocity = h[1]

        dt = 0.01
        mass = 1
        damping = w[1]
        spring_k = w[0]

        acceleration = u - damping * velocity - spring_k * position
        new_velocity = velocity + acceleration * dt
        new_position = position + velocity * dt

        new_h = np.array([new_position, new_velocity])

        if use_nonlin:
            return np.tanh(new_position), new_h
        return new_position, new_h
    return (rnn_step,)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create interactive elements
    weight_slider = mo.ui.slider(
        start=-1.5, stop=1.5, step=0.01, value=0.5, label="Recurrent Weight (w[0])"
    )
    second_weight_slider = mo.ui.slider(
        start=-1.5,
        stop=1.5,
        step=0.01,
        value=0.5,
        label="Second recurrent Weight (w[1])",
    )
    third_weight_slider = mo.ui.slider(
        start=-1.5,
        stop=1.5,
        step=0.01,
        value=0.5,
        label="Second recurrent Weight (w[2])",
    )
    input_slider = mo.ui.slider(
        start=-2.0, stop=2.0, step=0.1, value=1.0, label="Input Signal (u)"
    )
    initial_state_slider = mo.ui.slider(
        start=-2.0, stop=2.0, step=0.1, value=1.0, label="Initial State h(0)"
    )
    use_tanh = mo.ui.checkbox(False, label="Use tanh non-linearity")

    mo.hstack(
        [
            mo.vstack(
                [
                    weight_slider,
                    second_weight_slider,
                    third_weight_slider,
                    input_slider,
                    initial_state_slider,
                    use_tanh,
                ]
            ),
        ]
    )
    return (
        go,
        initial_state_slider,
        input_slider,
        make_subplots,
        mo,
        np,
        second_weight_slider,
        third_weight_slider,
        use_tanh,
        weight_slider,
    )


@app.cell
def _(
    generate_time_series,
    initial_state_slider,
    input_slider,
    second_weight_slider,
    third_weight_slider,
    use_tanh,
    visualize_rnn,
    weight_slider,
):
    # Usage:
    fig = visualize_rnn(
        weight_slider,
        second_weight_slider,
        third_weight_slider,
        input_slider,
        initial_state_slider,
        use_tanh,
        generate_time_series,
    )
    fig.show()
    return (fig,)


@app.cell
def _(np, rnn_step):
    # Cell for generating time series data
    def generate_time_series(w, u, h0, use_nonlin, steps=5000):
        h = h0
        time_points = np.arange(steps)
        outputs = []
        inputs = [u] * steps

        for _ in time_points:
            output, h = rnn_step(h, w, u, use_nonlin)
            outputs.append(output)

        return time_points, inputs, outputs
    return (generate_time_series,)


@app.cell
def _(go, np):
    def visualize_rnn(
        weight_slider,
        second_weight_slider,
        third_weight_slider,
        input_slider,
        initial_state_slider,
        use_tanh,
        generate_time_series,
    ):
        # Get parameters
        w = np.array(
            [
                weight_slider.value,
                second_weight_slider.value,
                third_weight_slider.value,
            ]
        )
        u = input_slider.value
        h0 = np.array(
            [
                initial_state_slider.value,
                0,
                0,
            ]
        )

        nonlin = use_tanh.value

        # Generate time series data
        time_points, inputs, outputs = generate_time_series(w, u, h0, nonlin)

        # Create figure
        fig = go.Figure()

        # Add time series traces
        fig.add_trace(
            go.Scatter(
                x=time_points, y=inputs, name="Input", line=dict(color="#1D84B5")
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_points, y=outputs, name="Output", line=dict(color="#E6425E")
            )
        )

        # Update layout
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=400,
            showlegend=True,
            title_text=f"Simple RNN (h(t) = {'tanh(' if nonlin else ''}w * h(t-1) + u{')'if nonlin else ''})",
            xaxis=dict(title="Time Steps", showgrid=True, gridcolor="lightgray"),
            yaxis=dict(title="Value", showgrid=True, gridcolor="lightgray"),
        )

        return fig
    return (visualize_rnn,)


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
