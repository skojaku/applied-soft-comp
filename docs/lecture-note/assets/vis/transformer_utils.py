import marimo as mo
import numpy as np
import pandas as pd
import altair as alt


def scatter_plot(
    df,
    df_original,
    color="#ff7f0e",
    width=300,
    height=300,
    size=100,
    title=None,
    vmax=2,
):
    """Generates an Altair scatter plot with word labels."""
    if vmax is None:
        vmax = np.maximum(np.max(np.abs(df["x"])), np.max(np.abs(df["y"])))

    base_original = (
        alt.Chart(df_original)
        .mark_circle(size=size, color="#dadada", opacity=0.8)
        .encode(
            x=alt.X("x", scale=alt.Scale(domain=[-vmax, vmax])),
            y=alt.Y("y", scale=alt.Scale(domain=[-vmax, vmax])),
            tooltip=["word"],
        )
    )
    base = (
        alt.Chart(df)
        .mark_circle(size=size, color=color)
        .encode(
            x=alt.X("x", scale=alt.Scale(domain=[-vmax, vmax])),
            y=alt.Y("y", scale=alt.Scale(domain=[-vmax, vmax])),
            tooltip=["word"],
        )
    )

    # Add vectors from origin to points with arrow heads
    vectors = (
        alt.Chart(df)
        .mark_line(
            color=color,
            opacity=0.5,
        )
        .encode(
            x=alt.X("x0:Q", scale=alt.Scale(domain=[-vmax, vmax])),
            x2=alt.X2("x:Q"),
            y=alt.Y("y0:Q", scale=alt.Scale(domain=[-vmax, vmax])),
            y2=alt.Y2("y:Q"),
            angle=alt.value(0),
        )
        .transform_calculate(
            x0="0",
            y0="0",  # Start at origin x=0  # Start at origin y=0
        )
    )

    base = base_original + base + vectors

    text = (
        alt.Chart(df)
        .mark_text(align="left", dx=10, dy=-5, fontSize=14)
        .encode(x="x", y="y", text="word")
    )

    return (base + text).properties(width=width, height=height, title=title)


def create_transformation_controls():
    """Creates the transformation control sliders."""
    x_scale = mo.ui.slider(
        0.1, 2.5, 0.1, value=1.0, label="$S_{\\text{x}}$", full_width=False
    )
    y_scale = mo.ui.slider(
        0.1, 2.5, 0.1, value=1.0, label="$S_{\\text{y}}$", full_width=False
    )
    rotation = mo.ui.slider(-180, 180, 1, value=0, label="$\\theta$", full_width=False)
    x_intercept = mo.ui.slider(
        -1, 1, 0.1, value=0, label="$b_{\\text{x}}$", full_width=False
    )
    y_intercept = mo.ui.slider(
        -1, 1, 0.1, value=0, label="$b_{\\text{y}}$", full_width=False
    )
    return x_scale, y_scale, rotation, x_intercept, y_intercept


def layout_controls(x_scale, y_scale, rotation, x_intercept, y_intercept):
    return mo.hstack(
        [
            mo.vstack([x_scale, y_scale, rotation]),
            mo.vstack([x_intercept, y_intercept]),
        ],
        align="start",
        widths=[0.3, 0.7],
        gap=0,
    )


def vertical_line():
    return mo.Html(
        "<div style='width: 2px; height: 100%; background-color: #ccc; margin: 0 10px;'></div>"
    )


def emb2df(words, embeddings, x_scale, y_scale, rotation, x_intercept, y_intercept):
    """Creates visualization with the current transformation values."""
    # Apply scaling
    scale_matrix = np.array([[x_scale, 0], [0, y_scale]])

    b = np.array([x_intercept, y_intercept])

    # Apply rotation
    theta = np.radians(rotation)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    # Compute transformed embeddings
    W = scale_matrix @ rotation_matrix
    transformed = embeddings @ W + b

    # Create dataframes
    original_df = pd.DataFrame(
        {"word": words, "x": embeddings[:, 0], "y": embeddings[:, 1]}
    )
    transformed_df = pd.DataFrame(
        {"word": words, "x": transformed[:, 0], "y": transformed[:, 1]}
    )
    return original_df, transformed_df, W, b


def heatmap(
    matrix,
    tick_labels=None,
    title=None,
    width=300,
    height=300,
    vmin=None,
    vmax=None,
    show_labels=True,
):
    # Convert matrix to DataFrame with explicit x, y, value columns
    data = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            data.append(
                {
                    "x": tick_labels[j] if tick_labels else j,
                    "y": tick_labels[i] if tick_labels else i,
                    "value": value,
                }
            )
    df = pd.DataFrame(data)

    base = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(
                "x", title=None, axis=alt.Axis(labelAngle=0) if show_labels else None
            ),
            y=alt.Y(
                "y", title=None, axis=alt.Axis(labelAngle=0) if show_labels else None
            ),
            color=alt.Color(
                "value",
                scale=alt.Scale(
                    scheme="blues", domain=[vmin, vmax] if vmin is not None else None
                ),
                legend=None,
            ),
            tooltip=["x", "y", "value"],
        )
    )

    text = (
        alt.Chart(df)
        .mark_text(baseline="middle")
        .encode(
            x="x",
            y="y",
            text=alt.Text("value", format=".2f"),
            color=alt.condition(
                alt.datum.value > (vmax if vmax else 1) / 2,
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    return (base + text).properties(width=width, height=height, title=title)
