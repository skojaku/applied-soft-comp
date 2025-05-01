import marimo

__generated_with = "0.12.5"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Chebyshev Polynomial and Graph Convolution

        ChebyNet constructs the adjacency matrix by summing up the Chebyshev polynomial of the normalized graph laplacian matrix:

        $$
        {\\bf L}_{\\text{learn}} = \\sum_{k=0}^{K-1} \\theta_k T_k(\\tilde{{\\bf L}})
        $$

        where $T_k$ is the Chebyshev polynomials:

        $$
        \\begin{align}
        T_0(x) &= 1\\\\ T_1(x) &= x \\\\ T_k(x) &= 2xT_{k-1}(x) - T_{k-2}(x)
        \\end{align}
        $$

        and $K$ is the degree of the polynomial, which is a hyperparameter. The normalized graph laplacian matrix is given by

        $$
        \\tilde {\\bf L} = {\\bf I} - {\\bf D}^{-1/2} {\\bf A} {\\bf D}^{-1/2}
        $$

        where $D$ is a diagonal matrix, with diagonal entries being the degrees of the nodes in the graph.

        ## A key benefit of Chebyshev polynomial

        Spectral Graph Convolution is computationally very expensive, as it involves the eigendecomposition of the $N \\times N$ graph lacplain matrix, which is ${\cal O}(N^3)$ time complexity and ${\cal O}(N^2)$ space complexity.

        The Chebyshev polynomial improves the computational cost by representing the convolution as a sum of $K$th order polynomials. This reduces the cost because the $k$th term $T_k$ involves the convolution between two nodes with distance up to $k$. Namely, $T_0$ defines the convolution between the nodes themselves, $T_1$ defines the convolution between the nodes with distance 1 (neighbors), and so on. By limiting the order $K$ to be small, the laplacin matrix can be computed with less computation time.


        Let us see the individual Chebyshev term $T_k$ in the heatmaps as follows.
        """
    )
    return


@app.cell
def _(binarize_switch, mo, term_slider):
    mo.vstack([term_slider, binarize_switch], align="start")
    return


@app.cell(hide_code=True)
def _(Tn, binarize_switch, np, plt, sns, term_slider):
    n_selected_terms = int(term_slider.value)

    ncols = np.minimum(n_selected_terms, 4)
    nrows = (n_selected_terms - 1) // 4 + 1

    height = 5 * nrows
    width = np.minimum(6.5 * ncols, 6.5 * 4)

    sns.set_style("white")
    sns.set(font_scale=1.2)
    sns.set_style("ticks")

    fig, axes = plt.subplots(figsize=(width, height), ncols=ncols, nrows=nrows)

    ax_flat = axes.flatten()
    for _i in range(n_selected_terms):
        ax = ax_flat[_i]
        _T = Tn[_i]

        if binarize_switch.value != 0:
            _T = np.array(_T != 0).astype(int)
            sns.heatmap(_T, ax=ax, cmap="Greys")
        else:
            sns.heatmap(_T, ax=ax, cmap="cividis")

    fig
    return (
        ax,
        ax_flat,
        axes,
        fig,
        height,
        n_selected_terms,
        ncols,
        nrows,
        width,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""The Chebyshev polynomial is efficient, namely it is the best polynimal representation in a sense of minimizing the maximum discrepancy between the data and polynomial functions. In the following visualization, we show a graph laplacian matrix and its Chebyshev approximation.""")
    return


@app.cell
def _(term_approx_slider):
    term_approx_slider
    return


@app.cell(hide_code=True)
def _(A, N, TargetMatrix, Tn, np, plt, sns, term_approx_slider, weights):
    n_selected_terms_approx = int(term_approx_slider.value)


    _fig, _axes = plt.subplots(figsize=(19.5, 5), ncols=3)

    S = A * 0
    for _i in range(n_selected_terms_approx):
        S += weights[_i] * Tn[_i]

    sns.set_style("white")
    sns.set(font_scale=1.4)
    sns.set_style("ticks")

    sns.heatmap(TargetMatrix, cmap="cividis", ax=_axes[0])
    sns.heatmap(S, cmap="cividis", ax=_axes[1])

    indices = np.triu_indices(N)
    s = S[indices]
    st = TargetMatrix[indices]
    # Define the limits
    lims = [
        min(min(s), min(st)),
        max(max(s), max(st)),
    ]

    _ax = sns.scatterplot(x=s, y=st, ax=_axes[2])
    _ax.set_xlabel("Approximated entry values")
    _ax.set_ylabel("True entry values")

    # Plot the equivalue (diagonal) line
    _ax.plot(lims, lims, "k--", lw=1.5, alpha=0.8)
    _ax.set_xlim(lims)
    _ax.set_ylim(lims)

    _axes[0].set_title("Target matrix")
    _axes[1].set_title("Chebyshev approximated")
    _axes[2].set_title(
        "Discrepancy between the true \n and approximated entry values"
    )
    sns.despine()

    _fig
    return S, indices, lims, n_selected_terms_approx, s, st


@app.cell(hide_code=True)
def _(mo, n_terms):
    term_slider = mo.ui.slider(
        start=2, stop=n_terms, value=2, label="Number of Chebyshev terms"
    )
    term_approx_slider = mo.ui.slider(
        start=2, stop=n_terms, value=2, label="Number of Chebyshev terms"
    )
    binarize_switch = mo.ui.switch(
        label="Display if entry values are zero or non-zero (non-zero shown in black)."
    )
    return binarize_switch, term_approx_slider, term_slider


@app.cell(hide_code=True)
def _(M, N, np):
    n_terms = 15
    weights = np.random.randn(n_terms) * np.exp(-1.2 * np.arange(n_terms))
    print(weights)
    Tn = [np.eye(N), M]
    TargetMatrix = M * 0
    for i in range(2, len(weights)):
        _tn = 2 * M @ Tn[-1] - Tn[-2]
        Tn.append(_tn)

    for i, tn in enumerate(Tn):
        TargetMatrix += weights[i] * tn
    return TargetMatrix, Tn, i, n_terms, tn, weights


@app.cell(hide_code=True)
def _(A, np):
    # Graph Laplacian
    N = A.shape[0]
    deg = np.array(A.sum(axis=1)).reshape(-1)
    Dsqrt = np.diag(1.0 / np.sqrt(deg))
    M = np.eye(N) - Dsqrt @ A @ Dsqrt
    return Dsqrt, M, N, deg


@app.cell(hide_code=True)
def _():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    G = nx.karate_club_graph()
    A = nx.to_scipy_sparse_array(G)
    return A, G, np, nx, plt, sns


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
