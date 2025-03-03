import marimo

__generated_with = "0.11.14-dev6"
app = marimo.App()


@app.cell
def _(mo):
    _fig = mo.md("![](https://github.com/user-attachments/assets/7f986426-a71f-444e-8e41-714a7cbe5add)")

    _text = mo.md("""
          # Transformers Inside Out

          <center>*An Interactive Guide to Attention, Residual Connections, and Positional Encoding*</center>
          <center>[@SadamoriKojaku](https://skojaku.github.io/)</center>
          """)
    mo.hstack([_text, _fig], align="center", widths=[0.5, 0.5], justify="center")
    return


@app.cell
def _(mo):
    mo.md("<br>" * 6)
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md(
                """
                - Many large language models (LLMs) are built based on a stack of transformer blocks.
                - Each transformer block takes a sequence of token vectors as input and outputs a sequence of token vectors (sequence-to-sequence!).
                - What is the trasnformer block?
                """
            ),
            mo.md(
                r"""
                 <img src="https://github.com/skojaku/applied-soft-comp/blob/master/docs/lecture-note/figs/transformer-overview.jpg?raw=true" width="250px">
        """
            ),
        ],
        align="center",
        widths=[1, 1],
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md(
                """
                - A transformer block consists of three components, i.e., multi-head attention, layer normalization, and feed-forward networks.
                - The key component is the **attention** module. """
            ),
            mo.md(
                r"""
                <img src="https://github.com/skojaku/applied-soft-comp/blob/master/docs/lecture-note/figs/transformer-component.jpg?raw=true" width="1000px">
        """
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md(
                """
                - An **attention module**:
                    - **Input**: A sequence of $N$ tokens
                    - **Output**: A transformed sequence of $N$ tokens

                - Each output token is *contextualized* by other tokens within the sequence
                - This allows the transformers to capture polysemies.

                - Let's look at each component one by one.

                """
            ),
            mo.md(
                r"""
                ![](https://github.com/skojaku/applied-soft-comp/blob/master/docs/lecture-note/figs/transformer-attention.jpg?raw=true)
        """
            ),
        ],
        align="center",
        widths=[1, 1],
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md(
                """
                - An attention module creates three types of vectors for each word:

                    - *Query vector* ,
                    - *Key vector*,
                    - *Value vector*

                - Each of these vectors are created by **a linear layer** ➡️.
                """
            ),
            mo.Html(
                r"""
                <img width="250" alt="Screenshot 2025-02-28 at 3 31 35 PM" src="https://github.com/user-attachments/assets/5f14c983-ced9-41b8-bab9-36e125ac708d" />
                """
            ),
        ],
        align="center",
        widths=[0.45, 0.55],
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(create_transformation_controls):
    # Create new controllers for this visualization
    linear_x_scale, linear_y_scale, linear_rotation, linear_x_intercept, linear_y_intercept = create_transformation_controls()
    return (
        linear_rotation,
        linear_x_intercept,
        linear_x_scale,
        linear_y_intercept,
        linear_y_scale,
    )


@app.cell
def _(
    emb2df,
    embeddings,
    layout_controls,
    linear_rotation,
    linear_x_intercept,
    linear_x_scale,
    linear_y_intercept,
    linear_y_scale,
    mo,
    scatter_plot,
    words,
):
    _original_df, _transformed_df, W, b = emb2df(
        words, embeddings, linear_x_scale.value, linear_y_scale.value, linear_rotation.value, linear_x_intercept.value, linear_y_intercept.value
    )

    _chart = scatter_plot(
        _transformed_df,
        _original_df,
        title="Linear Transformation",
        width=250,
        height=250,
        size=100,
    )

    # Create the visualization using current slider values
    # Create the UI controls
    mo.vstack(
        [
            mo.md("## Linear Transformation"),
            mo.hstack(
                [
                    _chart,
                    mo.vstack(
                        [
                            mo.md("""
                            Linear transformation:

                            $$
                            \\underbrace{\\begin{bmatrix}
                                x_{\\text{out}} & y_{\\text{out}}
                            \\end{bmatrix}}_{\\text{output}}
                            =
                            \\underbrace{\\begin{bmatrix}
                                x_{\\text{in}} & y_{\\text{in}}
                            \\end{bmatrix}}_{\\text{input}}
                            \\underbrace{\\begin{bmatrix}
                                w_{11} & w_{12} \\\\
                                w_{21} & w_{22} \\\\
                            \\end{bmatrix}}_{\\text{W}}+
                            \\underbrace{\\begin{bmatrix}
                                b_{1} \\\\
                                b_{2} \\\\
                            \\end{bmatrix}}_{\\text{b}}
                            $$
                            """),
                            mo.md(
                                """
                            $$
                            \\text{W} = \\begin{bmatrix}
                                %.2f & %.2f \\\\
                                %.2f & %.2f \\\\
                            \\end{bmatrix} +
                            \\begin{bmatrix}
                                %.2f \\\\
                                %.2f \\\\
                            \\end{bmatrix}
                            $$
                            """
                                % (W[0, 0], W[0, 1], W[1, 0], W[1, 1], b[0], b[1])
                            ),
                            layout_controls(linear_x_scale, linear_y_scale, linear_rotation, linear_x_intercept, linear_y_intercept),
                            mo.hstack(
                                [
                                    mo.md("""
                                          - $S_x$: X-axis scale
                                          - $S_y$: Y-axis scale
                                          - $\\theta$: Rotation
                                          """),
                                    mo.md("""
                                          - $b_x$: X-axis intercept
                                          - $b_y$: Y-axis intercept
                                          """),
                                ],
                            ),
                        ],
                    ),
                ], justify = "center", align="center",
            ),
        ]
    )
    return W, b


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md(
                """
                The query and key vectors are used to compute **the attention score**, i.e., how much each query word attends to key word, with a larger score indicating a stronger attendance.
                """
            ),
            mo.Html(
                r"""
                <img width="350" alt="Screenshot 2025-02-28 at 3 34 40 PM" src="https://github.com/user-attachments/assets/d33313c7-f11f-47eb-8a77-6fd78967bb47" />
                """
            ),
        ],
        align="center",
        justify="center",
        widths=[0.45, 0.55],
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(create_transformation_controls):
    # Create new controllers for query and key
    attention_query_x_scale, attention_query_y_scale, attention_query_rotation, attention_query_x_intercept, attention_query_y_intercept = create_transformation_controls()
    attention_key_x_scale, attention_key_y_scale, attention_key_rotation, attention_key_x_intercept, attention_key_y_intercept = create_transformation_controls()
    return (
        attention_key_rotation,
        attention_key_x_intercept,
        attention_key_x_scale,
        attention_key_y_intercept,
        attention_key_y_scale,
        attention_query_rotation,
        attention_query_x_intercept,
        attention_query_x_scale,
        attention_query_y_intercept,
        attention_query_y_scale,
    )


@app.cell
def _(
    attention_key_rotation,
    attention_key_x_intercept,
    attention_key_x_scale,
    attention_key_y_intercept,
    attention_key_y_scale,
    attention_query_rotation,
    attention_query_x_intercept,
    attention_query_x_scale,
    attention_query_y_intercept,
    attention_query_y_scale,
    emb2df,
    embeddings,
    heatmap,
    layout_controls,
    mo,
    np,
    scatter_plot,
    vertical_line,
    words,
):
    _transformed_df_key, _original_df_key, _W_key, _b_key = emb2df(
        words, embeddings, attention_key_x_scale.value, attention_key_y_scale.value, attention_key_rotation.value, attention_key_x_intercept.value, attention_key_y_intercept.value
    )
    _transformed_df_query, _original_df_query, _W_query, _b_query = emb2df(
        words,
        embeddings,
        attention_query_x_scale.value,
        attention_query_y_scale.value,
        attention_query_rotation.value,
        attention_query_x_intercept.value,
        attention_query_y_intercept.value,
    )

    _chart_key = scatter_plot(
        _original_df_key,
        _transformed_df_key,
        title="Key",
        width=170,
        height=170,
        size=100,
    )
    _chart_query = scatter_plot(
        _original_df_query,
        _transformed_df_query,
        title="Query",
        width=170,
        height=170,
        size=100,
    )

    _key_vecs = embeddings @ _W_key + _b_key
    _query_vecs = embeddings @ _W_query + _b_query
    _qk_matrix = _query_vecs @ _key_vecs.T


    _qk_matrix_normalized = np.exp(_qk_matrix / np.sqrt(_query_vecs.shape[1]))
    _qk_matrix_normalized = _qk_matrix_normalized / _qk_matrix_normalized.sum(
        axis=1, keepdims=True
    )

    _chart_qk_matrix = heatmap(
        _qk_matrix,
        tick_labels=words,
        title="QK matrix",
        width=300,
        height=300,
    )

    _chart_qk_matrix_normalized = heatmap(
        _qk_matrix_normalized,
        tick_labels=words,
        title="QK matrix",
        width=300,
        height=300,
        vmin=0,
    )

    # Create the visualization using current slider values
    # Create the UI controls
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(
                        """

                        ## Attention Score

                        The attention score is computed as follows:

                        $$
                        \\text{QK matrix} = Q K^\\top, \\quad
                        Q = \\begin{bmatrix}
                        q_1 \\\\
                        q_2 \\\\
                        \\vdots \\\\
                        q_N
                        \\end{bmatrix},\quad
                        K = \\begin{bmatrix}
                        k_1 \\\\
                        k_2 \\\\
                        \\vdots \\\\
                        k_N
                        \\end{bmatrix}
                        $$

                        $Q$ and $K$ are the matrices of query and key vectors, respectively.

                            """
                    ),
                    mo.hstack(
                        [
                            mo.vstack(
                                [
                                    layout_controls(attention_query_x_scale, attention_query_y_scale, attention_query_rotation, attention_query_x_intercept, attention_query_y_intercept),
                                    _chart_query,
                                ],
                                justify="center",
                                align="start",
                            ),
                            vertical_line(),
                            mo.vstack(
                                [
                                    layout_controls(attention_key_x_scale, attention_key_y_scale, attention_key_rotation, attention_key_x_intercept, attention_key_y_intercept),
                                    _chart_key,
                                ],
                                justify="center",
                                align="start",
                            ),
                        ]
                    ),
                ]
            ),
            mo.vstack(
                [
                    mo.md(
                        """
                        The resulting matrix has $(i,j)$-th entry $(QK^\\top)_{ij}$ which is the attention score between the $i$-th query and the $j$-th key.
                        """
                    ),
                    _chart_qk_matrix,
                ]
            ),
        ],
        align="center",
        justify="center",
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(create_transformation_controls):
    # Create new controllers for softmax visualization
    softmax_query_x_scale, softmax_query_y_scale, softmax_query_rotation, softmax_query_x_intercept, softmax_query_y_intercept = create_transformation_controls()
    softmax_key_x_scale, softmax_key_y_scale, softmax_key_rotation, softmax_key_x_intercept, softmax_key_y_intercept = create_transformation_controls()
    return (
        softmax_key_rotation,
        softmax_key_x_intercept,
        softmax_key_x_scale,
        softmax_key_y_intercept,
        softmax_key_y_scale,
        softmax_query_rotation,
        softmax_query_x_intercept,
        softmax_query_x_scale,
        softmax_query_y_intercept,
        softmax_query_y_scale,
    )


@app.cell
def _(
    emb2df,
    embeddings,
    heatmap,
    mo,
    np,
    scatter_plot,
    softmax_key_rotation,
    softmax_key_x_intercept,
    softmax_key_x_scale,
    softmax_key_y_intercept,
    softmax_key_y_scale,
    softmax_query_rotation,
    softmax_query_x_intercept,
    softmax_query_x_scale,
    softmax_query_y_intercept,
    softmax_query_y_scale,
    words,
):
    _transformed_df_key, _original_df_key, _W_key, _b_key = emb2df(
        words,
        embeddings,
        softmax_key_x_scale.value,
        softmax_key_y_scale.value,
        softmax_key_rotation.value,
        softmax_key_x_intercept.value,
        softmax_key_y_intercept.value,
    )
    _transformed_df_query, _original_df_query, _W_query, _b_query = emb2df(
        words,
        embeddings,
        softmax_query_x_scale.value,
        softmax_query_y_scale.value,
        softmax_query_rotation.value,
        softmax_query_x_intercept.value,
        softmax_query_y_intercept.value,
    )

    _chart_key = scatter_plot(
        _original_df_key,
        _transformed_df_key,
        title="Key",
        width=170,
        height=170,
        size=100,
    )
    _chart_query = scatter_plot(
        _original_df_query,
        _transformed_df_query,
        title="Query",
        width=170,
        height=170,
        size=100,
    )

    _key_vecs = embeddings @ _W_key + _b_key
    _query_vecs = embeddings @ _W_query + _b_query
    _qk_matrix = _query_vecs @ _key_vecs.T

    _qk_matrix_normalized = np.exp(_qk_matrix / np.sqrt(_query_vecs.shape[1]))
    _qk_matrix_normalized = _qk_matrix_normalized / _qk_matrix_normalized.sum(
        axis=1, keepdims=True
    )

    _chart_qk_matrix = heatmap(
        _qk_matrix,
        tick_labels=words,
        title="QK matrix",
        width=300,
        height=300,
    )

    _chart_qk_matrix_normalized = heatmap(
        _qk_matrix_normalized,
        tick_labels=words,
        title="QK matrix",
        width=300,
        height=300,
        vmin=0,
    )


    mo.hstack(
        [
            mo.md(
                """

               ##Normalizing Attention Score

               The attention score ranges between $-\\infty$ and $\\infty$, which is normalized into [0,1] by the softmax function **for each row**.

               $$
               \\text{Attention score} = \\text{softmax}\\left(\\frac{\\text{QK matrix}}{\\sqrt{d}}\\right)
               $$

               where $d$ is the dimension of the vectors.

               **The normalization ensures each row sums to 1.**
                """
            ),
            mo.ui.tabs(
                {
                    "QK matrix": _chart_qk_matrix,
                    "QK matrix (softmaxed)": _chart_qk_matrix_normalized,
                }
            ),
        ],
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    _question = mo.callout(mo.md("""**Question:** Why do we need the scaling factor $\\sqrt{d_k}$ in the attention score computation?

    $$
    \\text{Attention score} = \\text{softmax}\\left(\\frac{\\text{QK matrix}}{\\sqrt{d}}\\right)
    $$

    where $d$ is the dimension of the vectors.
                                 """))

    _answer = mo.accordion(
        {"Answer": """The scaling factor $\\sqrt{d_k}$, where $d_k$ is the dimensionality of the key vectors, is used to prevent the dot product of the query and key vectors from growing too large as $d_k$ increases. Without this scaling, larger values of $d_k$ could result in very large dot products, leading to extremely small gradients after applying the softmax function, which can slow down or hinder the training process. By scaling the dot product by $\\sqrt{d_k}$, the model maintains more stable gradients, leading to improved training stability and performance. :contentReference[oaicite:0]{index=0}"""
         }
    )
    mo.vstack([_question, _answer])
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(create_transformation_controls):
    # Create new controllers for QKV visualization
    qkv_query_x_scale, qkv_query_y_scale, qkv_query_rotation, qkv_query_x_intercept, qkv_query_y_intercept = create_transformation_controls()
    qkv_key_x_scale, qkv_key_y_scale, qkv_key_rotation, qkv_key_x_intercept, qkv_key_y_intercept = create_transformation_controls()
    qkv_value_x_scale, qkv_value_y_scale, qkv_value_rotation, qkv_value_x_intercept, qkv_value_y_intercept = create_transformation_controls()
    return (
        qkv_key_rotation,
        qkv_key_x_intercept,
        qkv_key_x_scale,
        qkv_key_y_intercept,
        qkv_key_y_scale,
        qkv_query_rotation,
        qkv_query_x_intercept,
        qkv_query_x_scale,
        qkv_query_y_intercept,
        qkv_query_y_scale,
        qkv_value_rotation,
        qkv_value_x_intercept,
        qkv_value_x_scale,
        qkv_value_y_intercept,
        qkv_value_y_scale,
    )


@app.cell
def _(
    emb2df,
    embeddings,
    heatmap,
    layout_controls,
    mo,
    np,
    pd,
    qkv_key_rotation,
    qkv_key_x_intercept,
    qkv_key_x_scale,
    qkv_key_y_intercept,
    qkv_key_y_scale,
    qkv_query_rotation,
    qkv_query_x_intercept,
    qkv_query_x_scale,
    qkv_query_y_intercept,
    qkv_query_y_scale,
    qkv_value_rotation,
    qkv_value_x_intercept,
    qkv_value_x_scale,
    qkv_value_y_intercept,
    qkv_value_y_scale,
    scatter_plot,
    words,
):
    _transformed_df_key, _original_df_key, _W_key, _b_key = emb2df(
        words,
        embeddings,
        qkv_key_x_scale.value,
        qkv_key_y_scale.value,
        qkv_key_rotation.value,
        qkv_key_x_intercept.value,
        qkv_key_y_intercept.value,
    )
    _transformed_df_query, _original_df_query, _W_query, _b_query = emb2df(
        words,
        embeddings,
        qkv_query_x_scale.value,
        qkv_query_y_scale.value,
        qkv_query_rotation.value,
        qkv_query_x_intercept.value,
        qkv_query_y_intercept.value,
    )
    _transformed_df_value, _original_df_value, _W_value, _b_value = emb2df(
        words,
        embeddings,
        qkv_value_x_scale.value,
        qkv_value_y_scale.value,
        qkv_value_rotation.value,
        qkv_value_x_intercept.value,
        qkv_value_y_intercept.value,
    )

    _width = 200
    _height = 200
    _chart_key = scatter_plot(
        _original_df_key,
        _transformed_df_key,
        title="Key",
        width=_width,
        height=_height,
        size=100,
    )
    _chart_query = scatter_plot(
        _original_df_query,
        _transformed_df_query,
        title="Query",
        width=_width,
        height=_height,
        size=100,
    )
    _chart_value = scatter_plot(
        _original_df_value,
        _transformed_df_value,
        title="Value",
        width=_width,
        height=_height,
        size=100,
    )

    _key_vecs = embeddings @ _W_key + _b_key
    _query_vecs = embeddings @ _W_query + _b_query
    _value_vecs = embeddings @ _W_value + _b_value
    _qk_matrix = _query_vecs @ _key_vecs.T

    _qk_matrix_normalized = np.exp(_qk_matrix / np.sqrt(_query_vecs.shape[1]))
    _qk_matrix_normalized = _qk_matrix_normalized / _qk_matrix_normalized.sum(
        axis=1, keepdims=True
    )

    _chart_qk_matrix_normalized = heatmap(
        _qk_matrix_normalized,
        tick_labels=words,
        title="QK matrix (Softmaxed)",
        width=200,
        height=200,
        vmin=0,
    )


    _output_vecs = _qk_matrix_normalized @ _value_vecs

    _transformed_df_output = pd.DataFrame(
        {"word": words, "x": _output_vecs[:, 0], "y": _output_vecs[:, 1]}
    )
    _original_df_output = pd.DataFrame(
        {"word": words, "x": embeddings[:, 0], "y": embeddings[:, 1]}
    )
    _chart_output = scatter_plot(
        _transformed_df_output,
        _original_df_output,
        title="V",
        width=300,
        height=300,
        size=100,
    )
    _description = mo.md(
        """
        ## Attention with QKV

        The output of the attention is computed as follows:

        $$
        \\text{Output} = \\text{softmax}\\left(\\frac{\\text{QK matrix}}{\\sqrt{d}}\\right) V
        $$

        where $V$ is the value vectors. This creates a weighted average of the value vector. For example, for the word "bank",

        $$
        \\text{v}^{\\text{out}}_{\\text{bank}} = {%.2f} v_{\\text{bank}} + {%.2f} v_{\\text{loan}} + {%.2f} v_{\\text{money}} + {%.2f} v_{\\text{river}} + {%.2f} v_{\\text{shore}}
        $$

        """
        % (
            _qk_matrix_normalized[0, 0],
            _qk_matrix_normalized[0, 1],
            _qk_matrix_normalized[0, 2],
            _qk_matrix_normalized[0, 3],
            _qk_matrix_normalized[0, 4],
        )
    )

    _query_plot = mo.vstack(
        [
            layout_controls(
                qkv_query_x_scale,
                qkv_query_y_scale,
                qkv_query_rotation,
                qkv_query_x_intercept,
                qkv_query_y_intercept,
            ),
            _chart_query,
        ],
        align="start",
    )

    key_plot = mo.vstack(
        [
            layout_controls(
                qkv_key_x_scale,
                qkv_key_y_scale,
                qkv_key_rotation,
                qkv_key_x_intercept,
                qkv_key_y_intercept,
            ),
            _chart_key,
        ],
        align="start",
    )

    mo.hstack(
        [
            mo.vstack([_description, mo.hstack([_query_plot, key_plot])]),
            mo.vstack(
                [
                    _chart_qk_matrix_normalized,
                    _chart_output,
                ]
            ),
        ],
        widths=[0.3, 0.6],
    )
    return (key_plot,)


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    _fig = mo.md(
        """
        ![](https://github.com/skojaku/applied-soft-comp/blob/master/docs/lecture-note/figs/transformer-multihead-attention.jpg?raw=true)
        """
    )

    _text = mo.md(
        """# Multi-Head Attention
        - Multi-head attention uses multiple attention heads to process input sequences in parallel.
        - Each head learns different contextualization of the input sequence.
            - e.g., one head learns how to contextualize for word 'bank' and another learns how to contextualize for word 'apple'.
        - The outputs from all heads are combined to create the final representation.
        """
    )

    mo.hstack(
        [
            _text,
            _fig,
        ],
        widths=[0.6, 0.4],
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(create_transformation_controls):
    # Create new controllers for each head
    # Head 1 UI elements
    head1_key_x_scale, head1_key_y_scale, head1_key_rotation, head1_key_x_intercept, head1_key_y_intercept = create_transformation_controls()
    head1_query_x_scale, head1_query_y_scale, head1_query_rotation, head1_query_x_intercept, head1_query_y_intercept = create_transformation_controls()
    head1_value_x_scale, head1_value_y_scale, head1_value_rotation, head1_value_x_intercept, head1_value_y_intercept = create_transformation_controls()

    # Head 2 UI elements
    head2_key_x_scale, head2_key_y_scale, head2_key_rotation, head2_key_x_intercept, head2_key_y_intercept = create_transformation_controls()
    head2_query_x_scale, head2_query_y_scale, head2_query_rotation, head2_query_x_intercept, head2_query_y_intercept = create_transformation_controls()
    head2_value_x_scale, head2_value_y_scale, head2_value_rotation, head2_value_x_intercept, head2_value_y_intercept = create_transformation_controls()

    num_heads = 2
    return (
        head1_key_rotation,
        head1_key_x_intercept,
        head1_key_x_scale,
        head1_key_y_intercept,
        head1_key_y_scale,
        head1_query_rotation,
        head1_query_x_intercept,
        head1_query_x_scale,
        head1_query_y_intercept,
        head1_query_y_scale,
        head1_value_rotation,
        head1_value_x_intercept,
        head1_value_x_scale,
        head1_value_y_intercept,
        head1_value_y_scale,
        head2_key_rotation,
        head2_key_x_intercept,
        head2_key_x_scale,
        head2_key_y_intercept,
        head2_key_y_scale,
        head2_query_rotation,
        head2_query_x_intercept,
        head2_query_x_scale,
        head2_query_y_intercept,
        head2_query_y_scale,
        head2_value_rotation,
        head2_value_x_intercept,
        head2_value_x_scale,
        head2_value_y_intercept,
        head2_value_y_scale,
        num_heads,
    )


@app.cell
def _(
    emb2df,
    embeddings,
    head1_key_rotation,
    head1_key_x_intercept,
    head1_key_x_scale,
    head1_key_y_intercept,
    head1_key_y_scale,
    head1_query_rotation,
    head1_query_x_intercept,
    head1_query_x_scale,
    head1_query_y_intercept,
    head1_query_y_scale,
    head1_value_rotation,
    head1_value_x_intercept,
    head1_value_x_scale,
    head1_value_y_intercept,
    head1_value_y_scale,
    head2_key_rotation,
    head2_key_x_intercept,
    head2_key_x_scale,
    head2_key_y_intercept,
    head2_key_y_scale,
    head2_query_rotation,
    head2_query_x_intercept,
    head2_query_x_scale,
    head2_query_y_intercept,
    head2_query_y_scale,
    head2_value_rotation,
    head2_value_x_intercept,
    head2_value_x_scale,
    head2_value_y_intercept,
    head2_value_y_scale,
    heatmap,
    layout_controls,
    mo,
    np,
    pd,
    scatter_plot,
    words,
):
    def process_head(
        head_id,
        words,
        embeddings,
        query_x_scale,
        query_y_scale,
        query_rotation,
        query_x_intercept,
        query_y_intercept,
        key_x_scale,
        key_y_scale,
        key_rotation,
        key_x_intercept,
        key_y_intercept,
        value_x_scale,
        value_y_scale,
        value_rotation,
        value_x_intercept,
        value_y_intercept,
        _width=200,
        _height=200,
    ):
        """Process QKV transformations and visualizations for a single head"""
        head_data = {}

        # Create transformed dataframes for key, query, value
        component_params = {
            "key": (
                key_x_scale,
                key_y_scale,
                key_rotation,
                key_x_intercept,
                key_y_intercept,
            ),
            "query": (
                query_x_scale,
                query_y_scale,
                query_rotation,
                query_x_intercept,
                query_y_intercept,
            ),
            "value": (
                value_x_scale,
                value_y_scale,
                value_rotation,
                value_x_intercept,
                value_y_intercept,
            ),
        }

        for component, (
            x_scale,
            y_scale,
            rotation,
            x_intercept,
            y_intercept,
        ) in component_params.items():
            _transformed_df, _original_df, _W, _b = emb2df(
                words,
                embeddings,
                x_scale.value,
                y_scale.value,
                rotation.value,
                x_intercept.value,
                y_intercept.value,
            )

            # Store results
            head_data[f"transformed_df_{component}"] = _transformed_df
            head_data[f"original_df_{component}"] = _original_df
            head_data[f"W_{component}"] = _W
            head_data[f"b_{component}"] = _b

            # Create scatter plot
            head_data[f"chart_{component}"] = scatter_plot(
                _original_df,
                _transformed_df,
                title=component.capitalize(),
                width=_width,
                height=_height,
                size=100,
                vmax=None,
            )

        # Compute attention mechanism
        _query_vecs = embeddings @ head_data["W_query"] + head_data["b_query"]
        _key_vecs = embeddings @ head_data["W_key"] + head_data["b_key"]
        _value_vecs = embeddings @ head_data["W_value"] + head_data["b_value"]

        _qk_matrix = _query_vecs @ _key_vecs.T
        _qk_matrix_normalized = np.exp(_qk_matrix / np.sqrt(_query_vecs.shape[1]))
        _qk_matrix_normalized = _qk_matrix_normalized / _qk_matrix_normalized.sum(
            axis=1, keepdims=True
        )

        head_data["qk_matrix_normalized"] = _qk_matrix_normalized
        head_data["chart_qk_matrix"] = heatmap(
            _qk_matrix_normalized,
            tick_labels=words,
            title="QK matrix (Softmaxed)",
            width=200,
            height=200,
            vmin=0,
            show_labels=False,
        )

        # Compute output vectors
        _output_vecs = _qk_matrix_normalized @ _value_vecs
        _transformed_df_output = pd.DataFrame(
            {"word": words, "x": _output_vecs[:, 0], "y": _output_vecs[:, 1]}
        )
        _original_df_output = pd.DataFrame(
            {"word": words, "x": embeddings[:, 0], "y": embeddings[:, 1]}
        )

        head_data["output_vecs"] = _output_vecs
        head_data["chart_output"] = scatter_plot(
            _transformed_df_output,
            _original_df_output,
            title="V",
            width=300,
            height=300,
            size=100,
            vmax=None,
        )

        return head_data


    def create_head_layout(
        head_id,
        head_data,
        query_x_scale,
        query_y_scale,
        query_rotation,
        query_x_intercept,
        query_y_intercept,
        key_x_scale,
        key_y_scale,
        key_rotation,
        key_x_intercept,
        key_y_intercept,
        value_x_scale,
        value_y_scale,
        value_rotation,
        value_x_intercept,
        value_y_intercept,
        description,
    ):
        """Create the layout for a single head's visualization"""
        controller = mo.vstack(
            [
                layout_controls(
                    query_x_scale,
                    query_y_scale,
                    query_rotation,
                    query_x_intercept,
                    query_y_intercept,
                ),
                head_data["chart_query"],
            ],
            align="start",
        )
        row_1 = mo.vstack([description, controller], align = "center")
        row_2 = mo.vstack([head_data["chart_output"], head_data["chart_qk_matrix"]], align = "center")
        return mo.hstack([row_1, row_2], align="center")


    def visualize_multihead_attention(words, embeddings, num_heads=2):
        """Create a complete multihead attention visualization with tabs for each head"""
        # Process all heads
        _all_head_data = {}
        _head_layouts = {}

        description_head_1 = mo.md(
            """
    ### 🏦 Financial "Bank" Attention

    Adjust the query vectors to help the model understand "bank" in a financial context. Try to make the attention focus on finance-related words like "money", "account", and "loan".

    Note: Only query vectors can be modified. Key and value vectors are fixed.
    """
        )
        description_head_2 = mo.md(
            """
    ### 🌊 River "Bank" Attention

    Adjust the query vectors to help the model understand "bank" in a geographical context. Make the attention focus on nature-related words like "river", "water", and "shore".

    Note: Only query vectors can be modified. Key and value vectors are fixed.
    """
        )
        # Head 1
        _all_head_data[1] = process_head(
            1,
            words,
            embeddings,
            head1_query_x_scale,
            head1_query_y_scale,
            head1_query_rotation,
            head1_query_x_intercept,
            head1_query_y_intercept,
            head1_key_x_scale,
            head1_key_y_scale,
            head1_key_rotation,
            head1_key_x_intercept,
            head1_key_y_intercept,
            head1_value_x_scale,
            head1_value_y_scale,
            head1_value_rotation,
            head1_value_x_intercept,
            head1_value_y_intercept,
        )

        _head_layouts["Head 1"] = create_head_layout(
            1,
            _all_head_data[1],
            head1_query_x_scale,
            head1_query_y_scale,
            head1_query_rotation,
            head1_query_x_intercept,
            head1_query_y_intercept,
            head1_key_x_scale,
            head1_key_y_scale,
            head1_key_rotation,
            head1_key_x_intercept,
            head1_key_y_intercept,
            head1_value_x_scale,
            head1_value_y_scale,
            head1_value_rotation,
            head1_value_x_intercept,
            head1_value_y_intercept,
            description_head_1,
        )

        # Head 2
        _all_head_data[2] = process_head(
            2,
            words,
            embeddings,
            head2_query_x_scale,
            head2_query_y_scale,
            head2_query_rotation,
            head2_query_x_intercept,
            head2_query_y_intercept,
            head2_key_x_scale,
            head2_key_y_scale,
            head2_key_rotation,
            head2_key_x_intercept,
            head2_key_y_intercept,
            head2_value_x_scale,
            head2_value_y_scale,
            head2_value_rotation,
            head2_value_x_intercept,
            head2_value_y_intercept,
        )

        _head_layouts["Head 2"] = create_head_layout(
            2,
            _all_head_data[2],
            head2_query_x_scale,
            head2_query_y_scale,
            head2_query_rotation,
            head2_query_x_intercept,
            head2_query_y_intercept,
            head2_key_x_scale,
            head2_key_y_scale,
            head2_key_rotation,
            head2_key_x_intercept,
            head2_key_y_intercept,
            head2_value_x_scale,
            head2_value_y_scale,
            head2_value_rotation,
            head2_value_x_intercept,
            head2_value_y_intercept,
            description_head_2,
        )

        instruction = mo.md("""# Exercise: Multi-Head Attention

        - Create two attention heads to help the model understand "bank" 🏦 in a financial context or river context 🏞️.
        - Click the tabs to switch between the two heads.
        """)

        # Create tabs
        return mo.vstack([instruction, mo.ui.tabs(_head_layouts)], align="center")


    ## Define a set of words with two polysemous words
    # _words = [
    #    "bank",    # polysemous: financial institution or river side
    #    "apple",   # polysemous: fruit or technology company
    #    "money",
    #    "loan",
    #    "river",
    #    "shore",
    #    "fruit",
    #    "orange",
    #    "computer",
    #    "phone"
    # ]
    #
    ## Create simple 2D embeddings where polysemous words are positioned between their different meaning clusters
    # np.random.seed(42)  # For reproducibility
    #
    ## Base embeddings - deliberately positioned to show polysemy
    # _embeddings = np.array([
    #    [-0.0, -0.3],   # bank (between financial and river clusters)
    #    [-0.1, 0.6],    # apple (better positioned between fruit and tech clusters)
    #    [-0.8, -0.3],   # money
    #    [-0.7, -0.6],   # loan
    #    [0.7, -0.5],    # river
    #    [0.6, -0.7],    # shore
    #    [0.6, 0.6],     # fruit
    #    [0.8, 0.4],     # orange
    #    [-0.5, 0.7],    # computer
    #    [-0.7, 0.5]     # phone
    # ]) * 2

    # Use the UI elements created in the previous cell
    multihead_viz = visualize_multihead_attention(words, embeddings)
    multihead_viz  # Display the visualization
    return (
        create_head_layout,
        multihead_viz,
        process_head,
        visualize_multihead_attention,
    )


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    _question = mo.callout(mo.md("""**Question:** Can we directly interpret attention weights to conclude that one word refers to another? For instance, if the word "this" has a high attention weight with the word "cat" in a sentence, does it imply that "this" refers to "cat"?"""))

    _answer = mo.accordion(
        {"Answer": """Not necessarily. While high attention weights between words like "this" and "cat" might suggest a relationship, attention weights are influenced by various factors and do not always indicate direct referential connections. Relying solely on attention weights for interpretability can be misleading, as they may not accurately reflect linguistic relationships. For a more in-depth analysis, refer to the study ["Is Attention Interpretable?" by Serrano and Smith (2019)](https://aclanthology.org/P19-1282/) """
         }
    )
    mo.vstack([_question, _answer])
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    _fig = mo.md(
        """
        <img width="60%" alt="Screenshot 2025-02-28 at 3 34 40 PM" src="https://github.com/skojaku/applied-soft-comp/blob/master/docs/lecture-note/figs/transformer-encoder-decoder.jpg?raw=true" />
        """
    )
    _text= mo.md(
        """
        # Encoder and Decoder Transformers

        - Many LLM adopts an encoder-decoder architecture.
        - Two types of transformers: **encoder** and **decoder**.
        - Each transformer has a different structure.
        """
    )
    mo.hstack(
        [
            _text,
            _fig,
        ],
        widths=[0.5, 0.5], align= "center", justify="center"
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    _fig = mo.md(
        """
        ![](https://github.com/skojaku/applied-soft-comp/blob/master/docs/lecture-note/figs/transformer-encoder.jpg?raw=true)
        """
    )
    _text= mo.md(
        """
        # Encoder Transformer

        - Encode the input sequence.
        - Multi-head attention.
        - Two **Layer normalizations** and **Residual connections**.
        - Feed-forward network.
        """
    )
    mo.hstack(
        [
            _text,
            _fig,
        ],
        widths=[0.5, 0.5], align="center", justify="center"
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(create_transformation_controls):
    x_scale_first, y_scale_first, rotation_first, x_intercept_first, y_intercept_first = create_transformation_controls()
    return (
        rotation_first,
        x_intercept_first,
        x_scale_first,
        y_intercept_first,
        y_scale_first,
    )


@app.cell
def _(
    alt,
    emb2df,
    embeddings,
    mo,
    np,
    pd,
    rotation_first,
    words,
    x_intercept_first,
    x_scale_first,
    y_intercept_first,
    y_scale_first,
):
    _fig = mo.md(
        """
        ![](https://github.com/skojaku/applied-soft-comp/blob/master/docs/lecture-note/figs/transformer-component.jpg?raw=true)
        """
    )


    def create_chart(
        df,
        vmax,
        vmin,
        color_before,
        color_after,
        width=200,
        height=200,
        label_before="before",
        label_after="after",
    ):
        """Create a chart comparing before and after states with consistent styling"""
        # Create a legend data frame
        legend_df = pd.DataFrame(
            [
                {"state": label_before, "x": vmin, "y": vmax},
                {"state": label_after, "x": vmin, "y": vmax},
            ]
        )

        # Create legend chart
        legend = (
            alt.Chart(legend_df)
            .mark_circle(size=100)
            .encode(
                x=alt.X("x", scale=alt.Scale(domain=[vmin, vmax])),
                y=alt.Y("y", scale=alt.Scale(domain=[vmin, vmax])),
                color=alt.Color(
                    "state:N",
                    scale=alt.Scale(
                        domain=[label_before, label_after],
                        range=[color_before, color_after],
                    ),
                ),
                opacity=alt.value(0),  # Make legend points invisible but keep legend
            )
            .properties(width=width, height=height, title="Data after layer 1")
        )

        base = (
            alt.Chart(df)
            .mark_circle(size=100)
            .encode(
                x=alt.X("x_changed", scale=alt.Scale(domain=[vmin, vmax])),
                y=alt.Y("y_changed", scale=alt.Scale(domain=[vmin, vmax])),
                color=alt.value(color_after),
                tooltip=["word_id"],
            )
        )

        base2 = (
            alt.Chart(df)
            .mark_circle(size=100)
            .encode(
                x=alt.X("x_without_change", scale=alt.Scale(domain=[vmin, vmax])),
                y=alt.Y("y_without_change", scale=alt.Scale(domain=[vmin, vmax])),
                color=alt.value(color_before),
                tooltip=["word_id"],
            )
        )
        lines = (
            alt.Chart(df)
            .mark_rule()
            .encode(
                x=alt.X("x_changed", scale=alt.Scale(domain=[vmin, vmax])),
                y=alt.Y("y_changed", scale=alt.Scale(domain=[vmin, vmax])),
                x2="x_without_change",
                y2="y_without_change",
                detail="word_id",
                color=alt.value(color_before),
            )
        )
        return lines + base + base2 + legend


    np.random.seed(42)
    _layer_params = [
        {
            "x_scale": x_scale_first.value,
            "y_scale": y_scale_first.value,
            "rotation": rotation_first.value,
            "x_intercept": x_intercept_first.value,
            "y_intercept": y_intercept_first.value,
        },
        {
            "x_scale": np.random.rand() * 3,
            "y_scale": np.random.rand() * 3,
            "rotation": np.random.rand() * 360,
            "x_intercept": np.random.rand(),
            "y_intercept": np.random.rand(),
        },
    ]

    # Neural network without changed parameters
    _df_0_without_change, _df_1_without_change, _W_1_without_change, _b_1_without_change = (
        emb2df(words, embeddings, 1, 1, 0, 0, 0)
    )
    _, _df_2_without_change, _W_2_without_change, _b_2_without_change = emb2df(
        words, embeddings, **_layer_params[1]
    )

    _df_0_without_change["layer_id"] = 0
    _df_1_without_change["layer_id"] = 1
    _df_2_without_change["layer_id"] = 2
    _df_without_change = pd.concat(
        [_df_0_without_change, _df_1_without_change, _df_2_without_change]
    )

    # Neural network with changed parameters
    _df_0, _df_1, _W, _b = emb2df(words, embeddings, **_layer_params[0])
    _emb = embeddings @ _W + _b
    _, _df_2, _W_2, _b_2 = emb2df(words, _emb, **_layer_params[1])

    _df_0["layer_id"] = 0
    _df_1["layer_id"] = 1
    _df_2["layer_id"] = 2
    _df = pd.concat([_df_0, _df_1, _df_2])

    vmax_0 = np.max(_df_0["x"])
    vmax_0 = np.max([vmax_0, np.max(_df_0["y"])])
    vmin_0 = np.min(_df_0["x"])
    vmin_0 = np.min([vmin_0, np.min(_df_0["y"])])

    vmax_1 = np.max(_df_1["x"])
    vmax_1 = np.max([vmax_1, np.max(_df_1["y"])])
    vmin_1 = np.min(_df_1["x"])
    vmin_1 = np.min([vmin_1, np.min(_df_1["y"])])

    vmax_2 = np.max(_df_2["x"])
    vmax_2 = np.max([vmax_2, np.max(_df_2["y"])])
    vmin_2 = np.min(_df_2["x"])
    vmin_2 = np.min([vmin_2, np.min(_df_2["y"])])

    vmax_2 = np.maximum(vmax_2, vmax_1)
    vmin_2 = np.minimum(vmin_2, vmin_1)

    vmax_1 = np.maximum(vmax_0, vmax_1)
    vmin_1 = np.minimum(vmin_0, vmin_1)

    # Rename columns before concatenating to avoid duplicates
    _df_changed = _df.rename(
        columns={"x": "x_changed", "y": "y_changed", "word": "word_id", "layer_id": "layer"}
    )
    _df_unchanged = _df_without_change.rename(
        columns={
            "x": "x_without_change",
            "y": "y_without_change",
            "word": "word_id",
            "layer_id": "layer",
        }
    )
    _df_combined = pd.merge(_df_changed, _df_unchanged, on=["word_id", "layer"])

    # Filter data for layer 1
    _df_layer1 = _df_combined[_df_combined["layer"] == 1]

    # Create chart for first layer only
    _chart_layer1 = create_chart(_df_layer1, vmax_1, vmin_1, "#FFB6C1", "#FF0000")

    # Create chart showing only layer 2
    _df_layer2 = _df_combined[_df_combined["layer"] == 2]

    # Create legend data frame for layer 2
    _legend_df2 = pd.DataFrame(
        [
            {"state": "before", "x": vmin_2, "y": vmax_2},
            {"state": "after", "x": vmin_2, "y": vmax_2},
        ]
    )


    _legend2 = (
        alt.Chart(_legend_df2)
        .mark_circle(size=100)
        .encode(
            x=alt.X("x", scale=alt.Scale(domain=[vmin_2, vmax_2])),
            y=alt.Y("y", scale=alt.Scale(domain=[vmin_2, vmax_2])),
            color=alt.Color(
                "state:N",
                scale=alt.Scale(domain=["before", "after"], range=["#ADD8E6", "#0000FF"]),
            ),
            opacity=alt.value(0),
        )
        .properties(
            title="Data after layer 2",
            width=200,
            height=200,
        )
    )

    _chart_layer2 = (
        (
            alt.Chart(_df_layer2)
            .mark_circle(size=100)
            .encode(
                x=alt.X("x_changed", scale=alt.Scale(domain=[vmin_2, vmax_2])),
                y=alt.Y("y_changed", scale=alt.Scale(domain=[vmin_2, vmax_2])),
                color=alt.value("#0000FF"),
                tooltip=["word_id"],
            )
        )
        + (
            alt.Chart(_df_layer2)
            .mark_circle(size=100)
            .encode(
                x=alt.X("x_without_change", scale=alt.Scale(domain=[vmin_2, vmax_2])),
                y=alt.Y("y_without_change", scale=alt.Scale(domain=[vmin_2, vmax_2])),
                color=alt.value("#ADD8E6"),
                tooltip=["word_id"],
            )
        )
        + (
            alt.Chart(_df_layer2)
            .mark_rule()
            .encode(
                x=alt.X("x_changed", scale=alt.Scale(domain=[vmin_2, vmax_2])),
                y=alt.Y("y_changed", scale=alt.Scale(domain=[vmin_2, vmax_2])),
                x2="x_without_change",
                y2="y_without_change",
                detail="word_id",
                color=alt.value("#ADD8E6"),
            )
        )
        + _legend2
    ).properties(
        width=300,
        height=300,
    )

    _text = mo.md(
        """# Layer Normalization (1)
        - Addresses the issue of **internal covariate shift**.

        - Internal covariate shift occurs when updates in earlier layers change the distribution of activations, forcing later layers to readjust continuously.
        """
    )

    _neural_net = mo.md(
        """
         ![](https://github.com/user-attachments/assets/9c5d2919-e27d-4ebe-b989-e5000db8ceb0)
        """
    )
    print(_neural_net.text)
    _chart_layer1 = mo.hstack(
        [_chart_layer1, mo.vstack([x_scale_first, y_scale_first, rotation_first, x_intercept_first, y_intercept_first])]
    )
    mo.hstack(
        [mo.vstack([_text, _neural_net, _chart_layer1]), _chart_layer2],
        widths=[1.0, 1.0],
    )
    return create_chart, vmax_0, vmax_1, vmax_2, vmin_0, vmin_1, vmin_2


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(create_transformation_controls):
    x_scale_second, y_scale_second, rotation_second, x_intercept_second, y_intercept_second = create_transformation_controls()
    return (
        rotation_second,
        x_intercept_second,
        x_scale_second,
        y_intercept_second,
        y_scale_second,
    )


@app.cell
def _(
    beta_slider,
    emb2df,
    embeddings,
    gamma_slider,
    mo,
    np,
    pd,
    rotation_second,
    scatter_plot,
    words,
    x_intercept_second,
    x_scale_second,
    y_intercept_second,
    y_scale_second,
):
    _original_df, _transformed_df, _W, _b = emb2df(
        words,
        embeddings,
        x_scale_second.value,
        y_scale_second.value,
        rotation_second.value,
        x_intercept_second.value,
        y_intercept_second.value,
    )

    _emb = embeddings @ _W + _b
    _mu = np.mean(_emb, axis=1).reshape(-1, 1)
    _sigma2 = np.var(_emb, axis=1).reshape(-1, 1)

    _emb_norm = (
        gamma_slider.value
        * (_emb - _mu @ np.ones((1, _emb.shape[1])))
        / np.sqrt(_sigma2 @ np.ones((1, _emb.shape[1])) + 1e-1)
        + beta_slider.value
    )

    # Create dataframes
    _normalized_df = pd.DataFrame(
        {"word": words, "x": _emb_norm[:, 0], "y": _emb_norm[:, 1]}
    )
    _fig_scatter_plot = scatter_plot(
        _transformed_df,
        _original_df,
        title="Input embeddings",
        width=200,
        height=200,
    )

    _fig_scatter_plot_normalized = scatter_plot(
        _normalized_df,
        _transformed_df,
        title="Layer normalization",
        width=200,
        height=200,
    )

    _text = mo.md(
        """# Layer normalization (2)
        - A common technique to normalize the data $x$ from previous layers before feeding into the next layer.

            $$
            \\text{LN}(x) = \\textcolor{red}{\\gamma} \\cdot \\frac{x - \\textcolor{blue}{\\mu}}{\\sqrt{\\textcolor{blue}{\\sigma^2} + \\epsilon}} + \\textcolor{red}{\\beta}
            $$

            where:

            - $\\textcolor{blue}{\\mu}$, $\\textcolor{blue}{\\sigma^2}$: mean and variance of $x$
            - $\\textcolor{red}{\\gamma}, \\textcolor{red}{\\beta}$: learnable parameters
            - $\\epsilon$: small constant for numerical stability

         - Example:

            $$
            \\begin{aligned}
            \\mu_{\\text{bank}} &= \\frac{v_{x,\\text{bank}} + v_{y,\\text{bank}}}{2} ={%.2f}, \\\\ \\sigma^2_{\\text{bank}} &= \\frac{1}{2} \\sum_{z \\in \\{x,y\\}} (v_{z,\\text{bank}} - \\mu_{\\text{bank}})^2  ={%.2f}
            \\end{aligned}
            $$

            $$
            \\begin{aligned}
            \\text{LN}(v_{\\text{bank}}) &= {%.2f} \\cdot \\frac{[{%.2f}, {%.2f}] - {%.2f}}{\\sqrt{{%.2f} + 1e-1}} + {%.2f} \\\\
            &= [{%.2f}, {%.2f}]
            \\end{aligned}
            $$

        """ % (_mu[0], _sigma2[0], gamma_slider.value, _emb[0,0], _emb[0,1], _mu[0], _sigma2[0], beta_slider.value, _emb_norm[0,0], _emb_norm[0,1])

    )

    mo.hstack(
        [
            _text,
            mo.vstack(
                [
                    mo.hstack(
                        [
                            _fig_scatter_plot,
                            mo.vstack(
                                [x_scale_second, y_scale_second, rotation_second, x_intercept_second, y_intercept_second]
                            ),
                        ],
                        align="start",
                    ),
                    mo.hstack(
                        [
                            _fig_scatter_plot_normalized,
                            mo.vstack([gamma_slider, beta_slider]),
                        ],
                        align="start",
                    ),
                ],
                align="start",
            ),
        ],
        widths=[0.6, 0.4],
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    _question = mo.callout(mo.md(
        """
        **Question:** Why is layer normalization preferred over standard normalization techniques, such as whitening (decorrelating inputs using the inverse covariance matrix) or scaling features to unit variance, in transformer architectures? Additionally, what potential issue arises from normalizing data by its norm?
        """
    ))

    _answer = mo.accordion(
        {
            "Answer": """
            Layer normalization is favored in transformer architectures because it normalizes across the features within a single data sample, making it *independent of batch size and sequence length*. This is particularly beneficial for models dealing with varying sequence lengths, as it ensures consistent normalization across different layers.

            In contrast, whitening involves computing the inverse covariance matrix to decorrelate input features, which is computationally intensive and can be impractical for large-scale models.

            Simply scaling features by its norm can destabilize gradients. Specifically, dividing data by its norm can lead to exploding or vanishing gradients during backpropagation, hindering effective training. See [python - Why is Normalization causing my network to have exploding gradients in training? - Stack Overflow](https://stackoverflow.com/questions/68122785/why-is-normalization-causing-my-network-to-have-exploding-gradients-in-training).
            """
        }
    )
    mo.vstack([_question, _answer])
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(create_transformation_controls):
    # Create new controllers specifically for residual connection visualization
    residual_x_scale, residual_y_scale, residual_rotation, residual_x_intercept, residual_y_intercept = create_transformation_controls()
    return (
        residual_rotation,
        residual_x_intercept,
        residual_x_scale,
        residual_y_intercept,
        residual_y_scale,
    )


@app.cell
def _(
    create_chart,
    emb2df,
    embeddings,
    layout_controls,
    mo,
    pd,
    residual_rotation,
    residual_x_intercept,
    residual_x_scale,
    residual_y_intercept,
    residual_y_scale,
    words,
):
    _text = mo.md(
        """# Residual connection

        ![](https://upload.wikimedia.org/wikipedia/commons/b/ba/ResBlock.png)

        A common technique to stabilize the training of deep neural networks.

    $$
    \\text{RC}(x) = x + \\textcolor{red}{F(x)}
    $$

    - $\\textcolor{red}{F(x)}$ is a function (e.g., a neural network) that learns "residual" information.
    - Useful when the input data has a complex distribution.
    - Stabilizes the training of deep neural networks (We will touch this in the next module).

    """
    )

    _fig_residual = mo.md(
        """![](https://upload.wikimedia.org/wikipedia/commons/b/ba/ResBlock.png)"""
    )

    # Get transformed and original embeddings
    _original_df, _transformed_df, _W, _b = emb2df(
        words, embeddings, residual_x_scale.value, residual_y_scale.value, residual_rotation.value, residual_x_intercept.value, residual_y_intercept.value
    )

    # Create DataFrame for visualization
    _vis_df = pd.DataFrame({
        'word_id': _transformed_df['word'],
        'x_without_change': _original_df['x'],
        'y_without_change': _original_df['y'],
        'x_changed': _transformed_df['x'] + _original_df['x'],
        'y_changed': _transformed_df['y'] + _original_df['y']
    })

    # Create visualization
    _residual_plot = create_chart(
        _vis_df,
        vmax=4,
        color_before="#FFB6C1",
        color_after="#FF0000",
        width=300,
        height=300,
        label_before='x',
        label_after='RC(x)'
    )

    mo.hstack(
        [
            _text,
            mo.vstack([_residual_plot, layout_controls(residual_x_scale, residual_y_scale, residual_rotation, residual_x_intercept, residual_y_intercept)])
        ],
        widths=[0.7, 0.3]
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    _question = mo.callout(mo.md(
        """
        **Question:** Why is it beneficial to use residual connections (where we compute \( g(x) = f(x) + x \)) instead of having a neural network layer directly learn the complete transformation? For example, focusing on a linear transformation, why learn \( W \) in \( Wx + x \) rather than directly learning \( W_g \) in \( W_g x \) where \( W_g = W + I \)?
        """
    ))

    _answer = mo.accordion(
        {
            "Answer": """
            Residual connections provide significant benefits by adding the input x to the output f(x). **Easier optimization** is achieved because the network only needs to learn small adjustments to identity rather than complete transformations, which is particularly valuable when the optimal function is close to identity.

            Residual connections also **prevent degradation** in deep networks by allowing each layer to focus on learning incremental improvements instead of having to learn the entire transformation from scratch.

            Perhaps most importantly, they enable **better gradient flow** by creating shortcuts for gradients during backpropagation, which substantially reduces vanishing gradient problems and makes training much deeper networks possible.


            See [He et al. (2016)](https://arxiv.org/abs/1512.03385) for more details.

            """
        }
    )
    mo.vstack([_question, _answer])
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    _fig = mo.md(
        """
        ![](https://github.com/skojaku/applied-soft-comp/blob/master/docs/lecture-note/figs/transformer-encoder.jpg?raw=true)
        """
    )
    _text= mo.md(
        """
        # Encoder Transformer (Revisit)

        - Encode the input sequence.
        - Multi-head attention.
        - Two **Layer normalizations** and **Residual connections**.
        - Feed-forward network.
        """
    )
    mo.hstack(
        [
            _text,
            _fig,
        ],
        widths=[0.5, 0.5], align="center"
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    _fig = mo.md(
        """
        ![](https://github.com/skojaku/applied-soft-comp/blob/master/docs/lecture-note/figs/transformer-decoder.jpg?raw=true)
        """
    )
    _text= mo.md(
        """
        # Decoder Transformer

        - Decode the output sequence.
        - **Masked** multi-head attention.
        - **Cross-attention** with the encoder.
        """
    )
    mo.hstack(
        [
            _text,
            _fig,
        ],
        widths=[0.5, 0.5], align="center"
    )
    return


@app.cell
def _(mo):
    _text = mo.md(
        """# Masked Attention

        Consider translating "I love you" → "Je t'aime". Training can be:

        - **Sequential:** Each token is generated step by step.

            - Step 1: Input → "I love you". Output → "Je"
            - Step 2: Input → "I love you Je". Output → "t'aime"

        - **Parallel (Masked Attention):** The decoder predicts all tokens at once, masking future ones.
            - Input → "I love you [$\\color{red}{\\text{mask}}$] [$\\color{red}{\\text{mask}}$]". Output → "Je [$\\color{red}{\\text{mask}}$] [$\\color{red}{\\text{mask}}$]"
            - Input → "I love you Je [$\\color{red}{\\text{mask}}$]". Output → "t'aime [$\\color{red}{\\text{mask}}$]"

        where **[$\\color{red}{\\text{mask}}$]** indicates a masked token that does not attend to any other tokens.
        Masked attention prevents a token from attending to future tokens by modifying attention scores.
        """
    )




    _figs = mo.md(
        """
        ![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe1317a05-3542-4158-94bf-085109a5793a_1220x702.png)
        """
    )

    mo.hstack(
        [
            _text,
            _figs,
        ],
        widths=[0.5, 0.5], align="center"
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    _text = mo.md("""
    # Cross-attention

    - This module blends the information from the encoder and the decoder.
    - The key and value vectors are formed with the encoder's output, and the query vector is from the decoder.
          """)

    _fig = mo.md(
        """
        ![](https://skojaku.github.io/applied-soft-comp/_images/transformer-cross-attention.jpg)
        """
    )

    mo.hstack(
        [
            _text,
            _fig,
        ],
        widths=[0.5, 0.5], align="center"
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    _text = mo.md("""
    # Overview of the Transformer

    #### **Attention modules:**

    - Contextualize the token embeddings

      - Self-attention, Cross-attention, Masked attention

    #### **Residual connections:**

    - Stabilize the training of deep neural networks.

    #### **Layer normalization:**

    - Normalize the token embeddings.

    #### **Feed-forward network:**

    - A fully connected neural network.

    ## What is the **Positional encoding 🤔?**
    """)


    _fig = mo.md(
        """
        ![](https://d2l.ai/_images/transformer.svg)
        """
    )

    mo.hstack(
        [
            _text,
            _fig,
        ],
        widths=[0.5, 0.5], align="center"
    )
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    # Create a slider for controlling the number of positions in the outer product matrix
    seq_len = 30
    position_slider = mo.ui.slider(2, seq_len, 1, value=5, label="Position Progress", full_width=False)
    d_model_slider = mo.ui.slider(2, 100, 1, value=2, label="Dimension of the embedding", full_width=False)
    return d_model_slider, position_slider, seq_len


@app.cell
def _(alt, d_model_slider, mo, np, pd, position_slider, seq_len):
    # Create a function to generate positional encodings
    def get_positional_encoding(seq_len, d_model):
        # Initialize the positional encoding matrix
        pos_enc = np.zeros((seq_len, d_model))
        # Calculate positional encodings
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))

        return pos_enc

    # Parameters for visualization
    d_model = d_model_slider.value  # Using 2D for visualization

    # Generate positional encodings
    pos_enc = get_positional_encoding(seq_len, d_model)

    # Create a dataframe for visualization with a frame column for animation
    df = pd.DataFrame()
    for frame in range(seq_len):
        temp_df = pd.DataFrame(
            {
                "position": range(frame + 1),
                "x": pos_enc[: frame + 1, 0],
                "y": pos_enc[: frame + 1, 1],
                "frame": frame,
            }
        )
        df = pd.concat([df, temp_df])
        if frame == position_slider.value:
            break

    print(df)

    # Create animated scatter plot
    scatter = (
        alt.Chart(df)
        .mark_circle(size=100)
        .encode(
            x=alt.X("x", scale=alt.Scale(domain=[-1.1, 1.1])),
            y=alt.Y("y", scale=alt.Scale(domain=[-1.1, 1.1])),
            color=alt.Color("position:O", scale=alt.Scale(scheme="reds")),
            tooltip=["position", "x", "y"],
        )
        .properties(width=300, height=300, title="Positional Encoding in 2D")
    )

    # Create animated line connecting points in sequence
    line = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x="x",
            y="y",
            order="position",
            color=alt.value("gray"),
            opacity=alt.value(0.3),
            strokeWidth=alt.value(0.5),
        )
    )

    # Add animation by frame
    animation = scatter + line
    #    .add_params(
    #        alt.selection_point(
    #            fields=['frame'],
    #            bind=alt.binding_range(
    #                min=0,
    #                max=seq_len-1,
    #                step=1,
    #                name='Position Progress:'
    #            ),
    #            name='animation'
    #        )
    #    ).transform_filter(
    #        'datum.frame <= animation.frame'
    #    )

    # Create a static version showing the full pattern for reference
    static_df = pd.DataFrame(
        {"position": range(seq_len), "x": pos_enc[:, 0], "y": pos_enc[:, 1]}
    )

    static_scatter = (
        alt.Chart(static_df)
        .mark_circle(size=60, opacity=1.0)
        .encode(
            x=alt.X("x", scale=alt.Scale(domain=[-1.1, 1.1])),
            y=alt.Y("y", scale=alt.Scale(domain=[-1.1, 1.1])),
            color=alt.Color("position:O", scale=alt.Scale(scheme="viridis")),
            tooltip=["position", "x", "y"],
        )
    )

    static_line = (
        alt.Chart(static_df)
        .mark_line(opacity=0.3)
        .encode(x="x", y="y", order="position", color=alt.value("gray"))
    )

    # Combine animated and static versions
    chart = animation  # + static_line

    # Function to update the outer product matrix based on slider value
    def update_outer_product(num_positions):
        # Select positions based on slider value
        positions_subset = list(range(num_positions))
        subset_enc = pos_enc[positions_subset]

        # Calculate outer products between position vectors
        outer_products = []
        for i in range(len(positions_subset)):
            for j in range(len(positions_subset)):
                # Calculate dot product similarity
                similarity = np.dot(subset_enc[i], subset_enc[j])
                outer_products.append(
                    {
                        "pos_i": positions_subset[i],
                        "pos_j": positions_subset[j],
                        "similarity": similarity,
                    }
                )

        outer_df = pd.DataFrame(outer_products)

        # Create heatmap of outer products
        heatmap = (
            alt.Chart(outer_df)
            .mark_rect()
            .encode(
                x=alt.X("pos_i:O", title="Position i"),
                y=alt.Y("pos_j:O", title="Position j"),
                color=alt.Color("similarity:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=["pos_i", "pos_j", "similarity"],
            )
            .properties(
                width=250,
                height=250,
                title=f"Similarity between Position Embeddings (n={num_positions})",
            )
        )

        return heatmap

    # Explanation text
    text = mo.md(
        """
    # 🔢 Positional Encoding

    - 🧩 Transformers don't know token order by default (self-attention sees tokens as a set)

    - ➕ We add position info to each token embedding:
      $$\\text{Embedding} = \\text{Token Embedding} + \\text{Positional Encoding}$$

    - 📐 Using sine/cosine functions:
      $$PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}})$$
      $$PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d_{model}})$$

    - This makes nearby positions stay close together.

    - This lets the model use both content AND position when paying attention

    - ▶️ **Try the slider to watch positions being added!**

    - 🔍 The heatmap shows similarity between positions

    - 🎛️ Adjust positions in the matrix with the slider below
    """
    )

    # Display the visualization and explanation side by side
    mo.hstack(
        [
            text,
            mo.vstack(
                [
                    chart,
                    position_slider,
                    d_model_slider,
                    update_outer_product(position_slider.value),
                ]
            ),
        ],
        widths=[0.6, 0.4],
        align="center",
    )

    # Make the heatmap update when the slider changes
    # position_slider.on_change(lambda _: mo.output.replace(update_outer_product(position_slider.value)))
    return (
        animation,
        chart,
        d_model,
        df,
        frame,
        get_positional_encoding,
        line,
        pos_enc,
        scatter,
        static_df,
        static_line,
        static_scatter,
        temp_df,
        text,
        update_outer_product,
    )


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    beta_slider = mo.ui.slider(-1, 1, 0.1, value=0.5, label="$\\beta$", full_width=False)
    gamma_slider = mo.ui.slider(0, 2, 0.1, value=1.0, label="$\\gamma$", full_width=False)
    return beta_slider, gamma_slider


@app.cell
def _(create_transformation_controls):
    query_x_scale_qkv, query_y_scale_qkv, query_rotation_qkv, query_x_intercept_qkv, query_y_intercept_qkv = create_transformation_controls()
    key_x_scale_qkv, key_y_scale_qkv, key_rotation_qkv, key_x_intercept_qkv, key_y_intercept_qkv = create_transformation_controls()
    value_x_scale_qkv, value_y_scale_qkv, value_rotation_qkv, value_x_intercept_qkv, value_y_intercept_qkv = create_transformation_controls()
    return (
        key_rotation_qkv,
        key_x_intercept_qkv,
        key_x_scale_qkv,
        key_y_intercept_qkv,
        key_y_scale_qkv,
        query_rotation_qkv,
        query_x_intercept_qkv,
        query_x_scale_qkv,
        query_y_intercept_qkv,
        query_y_scale_qkv,
        value_rotation_qkv,
        value_x_intercept_qkv,
        value_x_scale_qkv,
        value_y_intercept_qkv,
        value_y_scale_qkv,
    )


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    return alt, mo, np, pd


@app.cell
def _(alt, mo, np, pd):
    def scatter_plot(
        df,
        df_original,
        color="#ff7f0e",
        width=300,
        height=300,
        size=100,
        title=None,
        vmax = 2,
    ):
        """Generates an Altair scatter plot with word labels."""
        if vmax is None:
            vmax = np.maximum(np.max(np.abs(df['x'])), np.max(np.abs(df['y'])))

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
                angle=alt.value(0)
            )
            .transform_calculate(
                x0="0",  # Start at origin x=0
                y0="0"  # Start at origin y=0
            )
        )

        base = base_original + base + vectors

        text = (
            alt.Chart(df)
            .mark_text(align="left", dx=10, dy=-5, fontSize=14)
            .encode(
                x="x", y="y", text="word")
        )

        return (base + text).properties(width=width, height=height, title=title)


    def create_transformation_controls():
        """Creates the transformation control sliders."""
        x_scale = mo.ui.slider(0.1, 2.5, 0.1, value=1.0, label="$S_{\\text{x}}$", full_width=False)
        y_scale = mo.ui.slider(0.1, 2.5, 0.1, value=1.0, label="$S_{\\text{y}}$", full_width=False)
        rotation = mo.ui.slider(-180, 180, 1, value=0, label="$\\theta$", full_width=False)
        x_intercept = mo.ui.slider(-1, 1, 0.1, value=0, label="$b_{\\text{x}}$", full_width=False)
        y_intercept = mo.ui.slider(-1, 1, 0.1, value=0, label="$b_{\\text{y}}$", full_width=False)
        return x_scale, y_scale, rotation, x_intercept, y_intercept

    def layout_controls(x_scale, y_scale, rotation, x_intercept, y_intercept):
        return mo.hstack([
            mo.vstack([x_scale, y_scale, rotation]),
            mo.vstack([x_intercept, y_intercept]),
        ], align="start", widths=[0.3, 0.7], gap=0)


    def vertical_line():
        return mo.Html("<div style='width: 2px; height: 100%; background-color: #ccc; margin: 0 10px;'></div>")


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
    return (
        create_transformation_controls,
        emb2df,
        layout_controls,
        scatter_plot,
        vertical_line,
    )


@app.cell
def _(alt, pd):
    def heatmap(matrix, tick_labels=None, title=None, width=300, height=300, vmin=None, vmax=None, show_labels=True):
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

        # Calculate domain based on actual data
        min_val = df["value"].min() if vmin is None else vmin
        max_val = df["value"].max() if vmax is None else vmax

        # Set symmetric domain for better color contrast
        domain = [min_val, max_val]

        base = alt.Chart(df).mark_rect(strokeWidth=1, stroke="white").encode(
            x=alt.X(
                "x:N",  # Changed to nominal type for text labels
                title="",
                axis=alt.Axis(
                    labelAngle=45
                ),  # Angled labels for better readability
                sort=None,
            ),
            y=alt.Y(
                "y:N",  # Changed to nominal type for text labels
                title="",
                sort=None,  # Prevents automatic sorting
            ),
            color=alt.Color(
                "value",
                scale=alt.Scale(domain=domain, scheme="inferno", clamp=True),
                legend=alt.Legend(title="Value", orient="right"),
            )
        )

        if show_labels:  # Add parameter to control label visibility
            text_layer = alt.Chart(df).mark_text(baseline="middle", align="center").encode(
                x="x:N",
                y="y:N",
                text=alt.Text("value:Q", format=".2f"),
                color=alt.condition(
                    alt.datum.value < (domain[1] + domain[0])/2,
                    alt.value("white"),
                    alt.value("black")
                )
            )
            return (base + text_layer).properties(width=width, height=height, title=title)

        return base.properties(width=width, height=height, title=title)
    return (heatmap,)


@app.cell
def _():
    # Create the UI controls for each transformation
    #key_x, key_y, key_r, key_x_intercept, key_y_intercept = create_transformation_controls()
    #query_x, query_y, query_r, query_x_intercept, query_y_intercept = create_transformation_controls()
    #value_x, value_y, value_r, value_x_intercept, value_y_intercept = create_transformation_controls()
    return


@app.cell
def _(np):
    ## Define a set of words with two polysemous words
    words = [
        "bank",    # polysemous: financial institution or river side
    #    "apple",   # polysemous: fruit or technology company
        "money",
        "loan",
        "river",
        "shore",
    #    "fruit",
    #    "orange",
    #    "computer",
    #    "phone"
    ]

    # Base embeddings - deliberately positioned to show polysemy
    embeddings = np.array([
        [-0.0, -0.3],   # bank (between financial and river clusters)
    #    [-0.1, 0.6],    # apple (better positioned between fruit and tech clusters)
        [-0.8, -0.3],   # money
        [-0.7, -0.6],   # loan
        [0.7, -0.5],    # river
        [0.6, -0.7],    # shore
    #    [0.6, 0.6],     # fruit
    #    [0.8, 0.4],     # orange
    #    [-0.5, 0.7],    # computer
    #    [-0.7, 0.5]     # phone
    ]) * 2
    return embeddings, words


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    _question = mo.callout(mo.md(
        """
        **Question:** In transformer architectures, why don't we simply add positional indices like "1", "2", "3", etc., directly to the word embeddings to encode positional information?
        """
    ))

    _answer = mo.accordion(
        {
            "Answer": """
            Directly adding positional indices to word embeddings is problematic for three key reasons:

            **Magnitude Disparity:** In long sequences, large position values can overwhelm the semantic content of word embeddings, causing the model to focus too much on position rather than meaning.

            **Poor Relative Positioning:** Simple indices don't effectively capture the relationships between positions in a sequence, whereas sinusoidal encodings help the model understand relative distances between tokens.

            **Inconsistent Normalization:** Attempting to normalize position indices creates inconsistencies across sequences of different lengths, making it difficult for the model to generalize.

            Instead, transformers use sophisticated encoding methods like sinusoidal functions that maintain consistency and better support the model's pattern recognition capabilities.

            For more details and intuitive visualizations, see [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding)

            <video controls>
              <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/you-could-have-designed-SOTA-positional-encoding/IntegerEncoding.mp4" type="video/mp4">
              Your browser does not support the video tag.
            </video>
            """
        }
    )
    mo.vstack([_question, _answer])
    return


@app.cell
def _(mo):
    mo.md("<br>" * 7)
    return


@app.cell
def _(mo):
    _text = mo.md("""
          # Further Readings

            - [Attention is All You Need](https://arxiv.org/abs/1706.03762)
            -  [3Blue1Brown - Visualizing Attention, a Transformer's Heart | Chapter 6, Deep Learning](https://www.3blue1brown.com/lessons/attention)
            - [Transformer Explainer: LLM Transformer Model Visually Explained](https://poloclub.github.io/transformer-explainer/)
            - [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
            - [Dodrio](https://poloclub.github.io/dodrio/)
            - [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding)

          """)
    _text
    return


if __name__ == "__main__":
    app.run()
