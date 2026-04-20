import marimo

__generated_with = "0.23.1"
app = marimo.App(
    width="full",
    layout_file="layouts/network_embedding_slides.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import igraph as ig
    import networkx as nx
    from pyvis.network import Network
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    return ig, mo, np, nx, plt, sns


@app.cell(hide_code=True)
def _(mo):
    with open("marimo_lecture_note_theme.css") as _f:
        _css = _f.read()
    mo.Html(f"<style>{_css}</style>")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Network Embedding 🌐

    **Module 8 — Advanced Topics in Network Science** · *Sadamori Kojaku*

    ---

    We will build from scratch: what a network *is*, how to represent it as data,
    and how linear algebra gives us a principled way to compress it into a low-dimensional geometry.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is a Network?

    A **network** is nodes + edges. Nodes are entities (people, proteins, pages). Edges are relationships.
    We encode it as an **adjacency matrix** $\mathbf{A}$:

    $$A_{ij} = \begin{cases}1 & \text{if } i \text{ and } j \text{ are connected}\\0 & \text{otherwise}\end{cases}$$
    """)
    return


@app.cell(hide_code=True)
def _(nx, plt, sns):
    _G_kc = nx.karate_club_graph()
    _pos = nx.spring_layout(_G_kc, seed=42)
    _faction = nx.get_node_attributes(_G_kc, "club")
    _colors = ["#4ecdc4" if _faction[v] == "Mr. Hi" else "#ff6b6b" for v in _G_kc.nodes()]

    fig_kc, axes_kc = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    # Network
    ax_net = axes_kc[0]
    nx.draw_networkx_edges(_G_kc, _pos, ax=ax_net, alpha=0.3, width=0.8)
    nx.draw_networkx_nodes(_G_kc, _pos, ax=ax_net, node_color=_colors, node_size=200)
    nx.draw_networkx_labels(_G_kc, _pos, ax=ax_net, font_size=7, font_color="w")
    ax_net.set_title("Karate Club Network  (teal = Mr. Hi,  red = Officer)", fontsize=11)
    ax_net.axis("off")

    # Adjacency matrix (first 15 nodes for readability)
    _A_sub = nx.to_numpy_array(_G_kc)[:15, :15]
    sns.heatmap(
        _A_sub,
        ax=axes_kc[1],
        cmap="Blues",
        cbar=False,
        linewidths=0.4,
        linecolor="#333",
        xticklabels=range(15),
        yticklabels=range(15),
    )
    axes_kc[1].set_title("Adjacency matrix A (nodes 0–14)", fontsize=11)
    axes_kc[1].tick_params(labelsize=8)

    plt.tight_layout()
    fig_kc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From Adjacency Matrix to Embedding

    **PCA:** given data $\mathbf{X}$, find directions of max variance — eigenvectors of $\mathbf{X}^\top\mathbf{X}$.

    **Same idea here:** find directions that best reconstruct $\mathbf{A}$ — eigenvectors of $\mathbf{A}$.

    Both compress high-dimensional structure into a handful of numbers per node.
    """)
    return


@app.cell(hide_code=True)
def _(ig, np):
    # Load Zachary's karate club network once — used throughout
    G = ig.Graph.Famous("Zachary")
    A = np.array(G.get_adjacency().data, dtype=float)
    n = A.shape[0]
    return A, G, n


@app.cell(hide_code=True)
def _(mo):
    d_slider = mo.ui.slider(1, 20, value=5, label="d (eigenvectors)")
    return (d_slider,)


@app.cell(hide_code=True)
def _(A, d_slider, mo, np, plt, sns):
    _spec_header = mo.md(r"""
    ## Spectral Reconstruction

    Compress $\mathbf{A}$ into $\mathbf{U} \in \mathbb{R}^{n \times d}$ minimizing:

    $$\min_{\mathbf{U}} \tfrac{1}{2}\|\mathbf{A} - \mathbf{U}\mathbf{U}^\top\|_F^2$$

    **Result:** the optimal $\mathbf{U}$ is the top-$d$ eigenvectors of $\mathbf{A}$.
    Move the slider to see reconstruction quality improve with $d$.
    """)

    d = d_slider.value

    eigvals_full, eigvecs_full = np.linalg.eigh(A)
    idx = np.argsort(eigvals_full)[::-1]
    eigvals_sorted = eigvals_full[idx]
    eigvecs_sorted = eigvecs_full[:, idx]

    A_recon = sum(eigvals_sorted[i] * np.outer(eigvecs_sorted[:, i], eigvecs_sorted[:, i]) for i in range(d))
    error = 0.5 * np.linalg.norm(A - A_recon, "fro") ** 2

    # Single combined figure: heatmaps (top) + error curve (bottom)
    from matplotlib.gridspec import GridSpec as _GS

    fig_combined = plt.figure(figsize=(13, 6.5))
    _gs = _GS(2, 3, figure=fig_combined, height_ratios=[3, 2], hspace=0.45, wspace=0.15)

    vmax = np.abs(A).max()
    for _col, (mat, title) in enumerate(
        zip(
            [A, A_recon, A - A_recon],
            [r"Original $\mathbf{A}$", f"Reconstructed (d={d})", f"Residual (err={error:.0f})"],
        )
    ):
        _ax = fig_combined.add_subplot(_gs[0, _col])
        sns.heatmap(
            mat,
            ax=_ax,
            cmap="coolwarm",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        _ax.set_title(title, fontsize=11)

    _ax_e = fig_combined.add_subplot(_gs[1, :])
    _d_vals = list(range(1, 25))
    _errs = [
        0.5
        * np.linalg.norm(
            A - sum(eigvals_sorted[i] * np.outer(eigvecs_sorted[:, i], eigvecs_sorted[:, i]) for i in range(dv)),
            "fro",
        )
        ** 2
        for dv in _d_vals
    ]
    _ax_e.plot(_d_vals, _errs, marker="o", ms=4, color="steelblue")
    _ax_e.axvline(d, color="crimson", linestyle="--", label=f"d={d}")
    _ax_e.set_xlabel("d")
    _ax_e.set_ylabel("Frobenius error")
    _ax_e.legend(fontsize=9)
    _ax_e.set_title("Reconstruction error vs d", fontsize=11)

    fig_combined.subplots_adjust(top=0.92, bottom=0.10, left=0.06, right=0.98, hspace=0.45, wspace=0.15)

    mo.vstack([_spec_header, d_slider, fig_combined])
    return


@app.cell(hide_code=True)
def _(mo):
    _step1 = mo.md(r"""
    ## Laplacian Eigenmap — Derivation

    **Setup.** The graph Laplacian is $\mathbf{L} = \mathbf{D} - \mathbf{A}$. We want to embed each node $i$ into
    a coordinate $\mathbf{y}_i \in \mathbb{R}^d$ such that neighbors in the graph are also neighbors in space.

    **Step 1 — write down the cost.** For a single coordinate $x \in \mathbb{R}^n$, penalize
    every connected pair by the square of their distance:

    $$\Phi(\mathbf{x}) = \frac{1}{2}\sum_{i,j} A_{ij}(x_i - x_j)^2$$

    **Step 2 — rewrite in matrix form.** Expand $(x_i - x_j)^2 = x_i^2 - 2x_ix_j + x_j^2$ and collect:

    $$\Phi(\mathbf{x}) = \mathbf{x}^\top \mathbf{D}\mathbf{x} - \mathbf{x}^\top \mathbf{A}\mathbf{x} = \mathbf{x}^\top(\mathbf{D}-\mathbf{A})\mathbf{x} = \mathbf{x}^\top \mathbf{L} \mathbf{x}$$

    So minimizing $\mathbf{x}^\top \mathbf{L} \mathbf{x}$ is exactly minimizing the sum of squared distances over all edges.

    **Step 3 — add a constraint.** Without it, $\mathbf{x} = \mathbf{0}$ wins trivially. We normalize by node degree:

    $$\min_{\mathbf{x}} \mathbf{x}^\top \mathbf{L} \mathbf{x}, \quad \text{subject to} \quad \mathbf{x}^\top \mathbf{D} \mathbf{x} = 1, \quad \mathbf{x} \perp \mathbf{D}\mathbf{1}$$

    The $\mathbf{x} \perp \mathbf{D}\mathbf{1}$ condition removes the trivial constant solution $\mathbf{x} \propto \mathbf{1}$.

    **Step 4 — apply Lagrange multipliers.** Differentiate $\mathbf{x}^\top\mathbf{L}\mathbf{x} - \lambda(\mathbf{x}^\top\mathbf{D}\mathbf{x}-1)$
    with respect to $\mathbf{x}$ and set to zero:

    $$\mathbf{L}\mathbf{x} = \lambda\,\mathbf{D}\mathbf{x} \quad \Longrightarrow \quad \mathbf{D}^{-1/2}\mathbf{L}\mathbf{D}^{-1/2}\,\tilde{\mathbf{x}} = \lambda\,\tilde{\mathbf{x}}$$

    This is the **generalized eigenvalue problem**. The substitution $\tilde{\mathbf{x}} = \mathbf{D}^{1/2}\mathbf{x}$ converts it
    into a standard eigenproblem for the **normalized Laplacian** $\tilde{\mathbf{L}} = \mathbf{D}^{-1/2}\mathbf{L}\mathbf{D}^{-1/2}$.

    **Step 5 — read off the solution.** The minimum cost is $\lambda_2$ (the smallest *non-zero* eigenvalue),
    achieved at the **Fiedler vector** $\mathbf{x}_2$. For $d$ dimensions use eigenvectors $\mathbf{x}_2, \ldots, \mathbf{x}_{d+1}$.
    We skip $\mathbf{x}_1 \propto \mathbf{1}$ because $\lambda_1 = 0$ and it encodes no structure.
    """)

    mo.vstack([_step1])
    return


@app.cell(hide_code=True)
def _(mo):
    k_slider = mo.ui.slider(2, 6, value=2, label="k (clusters)")
    return (k_slider,)


@app.cell(hide_code=True)
def _(A, G, k_slider, mo, n, np, plt):
    k = k_slider.value

    D_mat = np.diag(A.sum(axis=1))
    L = D_mat - A
    eigvals_L, eigvecs_L = np.linalg.eigh(L)
    fiedler_vecs = eigvecs_L[:, 1:k]

    from sklearn.cluster import KMeans

    labels = (
        (fiedler_vecs[:, 0] > 0).astype(int)
        if k == 2
        else KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(fiedler_vecs)
    )

    layout = G.layout("fr")
    coords = np.array(layout.coords)

    fig_clust, axes_clust = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_g = axes_clust[0]
    for e in G.get_edgelist():
        ax_g.plot([coords[e[0], 0], coords[e[1], 0]], [coords[e[0], 1], coords[e[1], 1]], "k-", alpha=0.15, lw=0.7)
    ax_g.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", vmin=0, vmax=k - 1, s=70, zorder=3)
    ax_g.set_title(f"Spectral clustering  k={k}", fontsize=11)
    ax_g.axis("off")

    ax_em = axes_clust[1]
    ax_em.scatter(eigvecs_L[:, 1], eigvecs_L[:, 2], c=labels, cmap="tab10", vmin=0, vmax=k - 1, s=70)
    for vi in range(n):
        ax_em.annotate(str(vi), (eigvecs_L[vi, 1], eigvecs_L[vi, 2]), fontsize=6, alpha=0.6)
    ax_em.set_xlabel(r"$\mathbf{x}_2$ (Fiedler vector)")
    ax_em.set_ylabel(r"$\mathbf{x}_3$")
    ax_em.set_title("Laplacian eigenmap — topology as geometry", fontsize=11)
    plt.tight_layout()

    mo.vstack([k_slider, fig_clust])
    return


@app.cell(hide_code=True)
def _(mo):
    _n2v_text = mo.vstack(
        [
            mo.md(r"""
    ## node2vec — Neural Embeddings

    **word2vec (skip-gram):** given a word $w_t$, predict its neighbors within window $c$:
    $$\mathcal{L}=\sum_t\sum_{0<|j|\le c}\log P(w_{t+j}\mid w_t), \qquad P(w_O\mid w_I)=\frac{\exp(\mathbf{v}_{w_O}^\top\mathbf{v}_{w_I})}{\sum_w\exp(\mathbf{v}_w^\top\mathbf{v}_{w_I})}$$
    Each word gets a vector $\mathbf{v}_w\in\mathbb{R}^d$. Co-occurring words end up close in space.
    **node2vec** replaces words with nodes and sentences with biased random walks, then runs skip-gram.
    """),
            mo.md(r"""
    **Biased transition** (walk at $v$, came from $t$):
    $$P(\text{next}=x\mid v,t)\propto\begin{cases}1/p & d(t,x)=0\;\text{(backtrack)}\\1 & d(t,x)=1\;\text{(shared neighbor)}\\1/q & d(t,x)=2\;\text{(explore further)}\end{cases}$$

    Low $p$ → BFS-like, stays local → captures **structural role**.
    Low $q$ → DFS-like, ventures far → captures **community membership**.
    """),
        ]
    )

    _walk_img_url = "https://snap.stanford.edu/node2vec/walk.png"
    _walk_fig = mo.vstack(
        [
            mo.Html(f'<img src="{_walk_img_url}" style="max-width:100%; border-radius:6px;" />'),
            mo.md("*Grover & Leskovec (2016), Fig. 1 — biased walk at node $v$ after visiting $t$.*"),
        ]
    )

    mo.hstack([_n2v_text, _walk_fig], widths=[3, 2], gap=40)
    return


@app.cell(hide_code=True)
def _(G, np):
    import random as _rnd2


    def biased_rw(graph, start, length, ret_p, inout_q, seed=None):
        if seed is not None:
            _rnd2.seed(seed)
        walk = [start]
        prev = None
        for _ in range(length - 1):
            cur = walk[-1]
            nbrs = graph.neighbors(cur)
            if not nbrs:
                break
            if prev is None:
                walk.append(_rnd2.choice(nbrs))
            else:
                wts = [
                    1.0 / ret_p if nb == prev else (1.0 if graph.are_adjacent(nb, prev) else 1.0 / inout_q) for nb in nbrs
                ]
                walk.append(_rnd2.choices(nbrs, weights=wts, k=1)[0])
            prev = cur
        return walk


    _rw_layout = G.layout("fr")
    rw_coords = np.array(_rw_layout.coords)
    return biased_rw, rw_coords


@app.cell(hide_code=True)
def _(mo, n):
    rw_start = mo.ui.slider(0, n - 1, value=0, label="Start node")
    rw_steps = mo.ui.slider(2, 35, value=12, label="Steps")
    rw_p = mo.ui.slider(0.25, 4.0, value=1.0, step=0.25, label="p")
    rw_q = mo.ui.slider(0.25, 4.0, value=1.0, step=0.25, label="q")
    return rw_p, rw_q, rw_start, rw_steps


@app.cell(hide_code=True)
def _(G, biased_rw, mo, n, plt, rw_coords, rw_p, rw_q, rw_start, rw_steps):
    _walk = biased_rw(G, rw_start.value, rw_steps.value, rw_p.value, rw_q.value, seed=0)
    _wset = set(_walk)

    fig_rw, ax_rw = plt.subplots(figsize=(7, 5.5))
    fig_rw.patch.set_facecolor("#f8f8f6")
    ax_rw.set_facecolor("#f8f8f6")

    for _bg_e in G.get_edgelist():
        ax_rw.plot(
            [rw_coords[_bg_e[0], 0], rw_coords[_bg_e[1], 0]],
            [rw_coords[_bg_e[0], 1], rw_coords[_bg_e[1], 1]],
            "-",
            color="#666",
            alpha=0.25,
            lw=0.8,
        )
    for _wi in range(len(_walk) - 1):
        _wu, _wv = _walk[_wi], _walk[_wi + 1]
        ax_rw.annotate(
            "",
            xy=(rw_coords[_wv, 0], rw_coords[_wv, 1]),
            xytext=(rw_coords[_wu, 0], rw_coords[_wu, 1]),
            arrowprops=dict(arrowstyle="-|>", color="#7a3535", lw=1.8),
        )
    # Desaturated colors: unvisited=slate-blue, visited=muted amber
    _nc = ["#5a7a8a" if _nv not in _wset else "#9a7040" for _nv in range(n)]
    ax_rw.scatter(rw_coords[:, 0], rw_coords[:, 1], c=_nc, s=90, zorder=3, edgecolors="none")
    # Start: dark teal star, End: dark olive diamond
    ax_rw.scatter(*rw_coords[_walk[0]], s=240, color="#2a6060", zorder=5, marker="*")
    ax_rw.scatter(*rw_coords[_walk[-1]], s=170, color="#5a6a2a", zorder=5, marker="D")
    for _nv in range(n):
        ax_rw.annotate(
            str(_nv), rw_coords[_nv], fontsize=6, ha="center", va="center", color="#f0eeea", fontweight="bold", zorder=6
        )
    ax_rw.set_title(f"Walk  p={rw_p.value}  q={rw_q.value}  ({rw_steps.value} steps)", fontsize=11, color="#333")
    ax_rw.axis("off")
    plt.tight_layout()

    _walk_str = " \u2192 ".join(str(_wv) for _wv in _walk)
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.hstack([rw_start, rw_steps]),
                    mo.hstack([rw_p, rw_q]),
                    mo.md(f"**Walk:** `{_walk_str}`"),
                ]
            ),
            fig_rw,
        ],
        widths=[1, 2],
        gap=24,
    )
    return


@app.cell(hide_code=True)
def _(G, biased_rw, mo, n, plt):
    import random as _plxj_rnd
    import numpy as _np2


    def _ppmi_embed(graph, n_nodes, p, q, n_walks=40, walk_len=40, window=5, d=2, seed=42):
        """PPMI + truncated SVD: the matrix that word2vec skip-gram implicitly factorizes."""
        _plxj_rnd.seed(seed)
        _walks = [biased_rw(graph, v, walk_len, p, q) for v in range(n_nodes) for _ in range(n_walks)]
        _cooc = _np2.zeros((n_nodes, n_nodes))
        for _walk in _walks:
            for _i, _u in enumerate(_walk):
                for _j in range(max(0, _i - window), min(len(_walk), _i + window + 1)):
                    if _i != _j:
                        _cooc[_u, _walk[_j]] += 1
        _N = _cooc.sum() + 1e-10
        _rs = _cooc.sum(axis=1, keepdims=True)
        _rs[_rs == 0] = 1
        _cs = _cooc.sum(axis=0, keepdims=True)
        _cs[_cs == 0] = 1
        _ppmi = _np2.maximum(_np2.log(_cooc * _N / (_rs * _cs) + 1e-10), 0)
        _U, _s, _ = _np2.linalg.svd(_ppmi, full_matrices=False)
        return _U[:, :d] * _np2.sqrt(_s[:d])


    _configs = [
        (0.25, 4.0, "p=0.25, q=4\n(BFS-like → structural role)"),
        (1.0, 1.0, "p=1, q=1\n(unbiased walk)"),
        (4.0, 0.25, "p=4, q=0.25\n(DFS-like → community)"),
    ]
    _gt = [0 if v <= 16 else 1 for v in range(n)]
    _node_colors = ["#4a6fa5" if c == 0 else "#b55a5a" for c in _gt]

    fig_n2v, _axes = plt.subplots(1, 3, figsize=(13, 4))
    fig_n2v.patch.set_facecolor("#f8f8f6")
    for _ax, (_p, _q, _label) in zip(_axes, _configs):
        _emb = _ppmi_embed(G, n, _p, _q)
        _ax.set_facecolor("#f8f8f6")
        _ax.scatter(_emb[:, 0], _emb[:, 1], c=_node_colors, s=70, zorder=3, edgecolors="none")
        for _v in range(n):
            _ax.annotate(str(_v), _emb[_v], fontsize=6, alpha=0.5, ha="center", va="center")
        _ax.set_title(_label, fontsize=10)
        _ax.axis("off")

    from matplotlib.patches import Patch as _Patch

    _legend_handles = [_Patch(color="#4a6fa5", label="Mr. Hi"), _Patch(color="#b55a5a", label="Officer")]
    _axes[1].legend(
        handles=_legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    plt.tight_layout()

    mo.vstack(
        [
            mo.md(
                r"""### How p and q Shape the Embedding

    With **DFS-like walks** (high $p$, low $q$) the walk roams freely through the graph and collects distant context.
    Nodes in the same *community* encounter each other often — so they cluster together.
    With **BFS-like walks** (low $p$, high $q$) the walk stays near its origin and repeatedly samples the local neighborhood.
    Nodes that play the same *structural role* — similar degree, similar local structure — end up close, even across different communities.
    """
            ),
            fig_n2v,
            mo.md("*Karate club network (Zachary 1977). Color marks faction membership.*"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Three families of network embedding, all rooted in linear algebra:

    **Spectral (adjacency)** — top eigenvectors of $\mathbf{A}$ minimize reconstruction error.

    **Laplacian / graph cuts** — smallest eigenvectors of $\mathbf{L}=\mathbf{D}-\mathbf{A}$ minimize the cut objective. Topology becomes Euclidean geometry.

    **Neural (node2vec)** — random walks + skip-gram. Learns flexible, non-linear embeddings. Parameters p and q trade off local vs. global structure.
    """)
    return


if __name__ == "__main__":
    app.run()
