import marimo as mo
import numpy as np
import pandas as pd
from transformer_utils import (
    scatter_plot,
    heatmap,
    create_transformation_controls,
    layout_controls,
    emb2df,
)


def get_data():
    words = [
        "bank",
        "money",
        "loan",
        "river",
        "shore",
    ]

    # Base embeddings
    embeddings = (
        np.array(
            [
                [0.0, 0.0],  # bank (center)
                [-0.8, -0.3],  # money
                [-0.7, -0.6],  # loan
                [0.7, -0.5],  # river
                [0.6, -0.7],  # shore
            ]
        )
        * 2
    )
    return embeddings, words


def visualize_static_embeddings():
    embeddings, words = get_data()
    _df = pd.DataFrame({"word": words, "x": embeddings[:, 0], "y": embeddings[:, 1]})

    _chart = scatter_plot(
        _df,
        _df,  # Pass same df as original for now
        title="Static Word Embeddings",
        width=400,
        height=400,
        size=100,
    )

    return mo.vstack(
        [mo.md("Here are our word vectors. Notice 'bank' is in the middle."), _chart],
        align="center",
    )


def visualize_contextualization():
    embeddings, words = get_data()

    # Sliders for each word
    slider_money = mo.ui.slider(0, 1, 0.01, value=0, label="Money Weight")
    slider_loan = mo.ui.slider(0, 1, 0.01, value=0, label="Loan Weight")
    slider_river = mo.ui.slider(0, 1, 0.01, value=0, label="River Weight")
    slider_shore = mo.ui.slider(0, 1, 0.01, value=0, label="Shore Weight")
    slider_bank = mo.ui.slider(0, 1, 0.01, value=1, label="Bank Weight")

    sliders = {
        "money": slider_money,
        "loan": slider_loan,
        "river": slider_river,
        "shore": slider_shore,
        "bank": slider_bank,
    }

    def compute_contextualized_vector(weights):
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight == 0:
            return np.zeros(2)

        weighted_sum = np.zeros(2)
        for i, word in enumerate(words):
            weighted_sum += embeddings[i] * (weights[word] / total_weight)
        return weighted_sum

    # QK Visualization
    _weights = {word: slider.value for word, slider in sliders.items()}
    _new_vec = compute_contextualized_vector(_weights)

    # Plot
    _df_orig = pd.DataFrame(
        {"word": words, "x": embeddings[:, 0], "y": embeddings[:, 1]}
    )
    _df_new = pd.DataFrame(
        {"word": ["Contextualized Bank"], "x": [_new_vec[0]], "y": [_new_vec[1]]}
    )

    _chart_orig = scatter_plot(
        _df_orig, _df_orig, color="#dadada", title="Original Words"
    )
    _chart_new = scatter_plot(
        _df_new, _df_orig, color="#ff7f0e", title="Contextualized Result"
    )

    return mo.hstack([mo.vstack(list(sliders.values())), _chart_new], align="center")


def visualize_attention_mechanism():
    embeddings, words = get_data()

    # Controls for Q and K
    q_x_s, q_y_s, q_rot, q_x_i, q_y_i = create_transformation_controls()
    k_x_s, k_y_s, k_rot, k_x_i, k_y_i = create_transformation_controls()

    # Transform
    _, _, W_q, b_q = emb2df(
        words,
        embeddings,
        q_x_s.value,
        q_y_s.value,
        q_rot.value,
        q_x_i.value,
        q_y_i.value,
    )
    _, _, W_k, b_k = emb2df(
        words,
        embeddings,
        k_x_s.value,
        k_y_s.value,
        k_rot.value,
        k_x_i.value,
        k_y_i.value,
    )

    Q = embeddings @ W_q + b_q
    K = embeddings @ W_k + b_k

    # Scores
    _scores = Q @ K.T

    # Normalize (Softmax)
    _exp_scores = np.exp(_scores - np.max(_scores, axis=1, keepdims=True))
    weights = _exp_scores / np.sum(_exp_scores, axis=1, keepdims=True)

    _df_q = pd.DataFrame({"word": words, "x": Q[:, 0], "y": Q[:, 1]})
    _df_k = pd.DataFrame({"word": words, "x": K[:, 0], "y": K[:, 1]})
    _df_orig = pd.DataFrame(
        {"word": words, "x": embeddings[:, 0], "y": embeddings[:, 1]}
    )

    chart_q = scatter_plot(
        _df_q, _df_orig, title="Query Vectors (Q)", width=200, height=200
    )
    chart_k = scatter_plot(
        _df_k, _df_orig, title="Key Vectors (K)", width=200, height=200
    )

    chart_weights = heatmap(
        weights,
        tick_labels=words,
        title="Attention Weights (Softmax)",
        width=300,
        height=300,
        vmin=0,
        vmax=1,
    )

    return mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("**Query Transformation**"),
                            layout_controls(q_x_s, q_y_s, q_rot, q_x_i, q_y_i),
                            chart_q,
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("**Key Transformation**"),
                            layout_controls(k_x_s, k_y_s, k_rot, k_x_i, k_y_i),
                            chart_k,
                        ]
                    ),
                ],
                align="center",
            ),
            mo.md("**Resulting Attention Weights:**"),
            chart_weights,
        ],
        align="center",
    )
