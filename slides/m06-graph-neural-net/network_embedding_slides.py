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
    The adjacency matrix $\mathbf{A}$ encodes it: $A_{ij}=1$ if $i$ and $j$ are connected, else 0.
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
    mo.md(r"""
    ## Spectral Reconstruction

    Compress $\mathbf{A}$ into $\mathbf{U} \in \mathbb{R}^{n \times d}$ minimizing:

    $$\min_{\mathbf{U}} \tfrac{1}{2}\|\mathbf{A} - \mathbf{U}\mathbf{U}^\top\|_F^2$$

    **Result:** the optimal $\mathbf{U}$ is the top-$d$ eigenvectors of $\mathbf{A}$.
    Move the slider to see reconstruction quality improve with $d$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    d_slider = mo.ui.slider(1, 20, value=5, label="d (eigenvectors)")
    return (d_slider,)


@app.cell(hide_code=True)
def _(A, d_slider, mo, np, plt, sns):
    d = d_slider.value

    eigvals_full, eigvecs_full = np.linalg.eigh(A)
    idx = np.argsort(eigvals_full)[::-1]
    eigvals_sorted = eigvals_full[idx]
    eigvecs_sorted = eigvecs_full[:, idx]

    A_recon = sum(eigvals_sorted[i] * np.outer(eigvecs_sorted[:, i], eigvecs_sorted[:, i]) for i in range(d))
    error = 0.5 * np.linalg.norm(A - A_recon, "fro") ** 2

    # ---- row 1: heatmaps ----
    fig_r, axes_r = plt.subplots(1, 3, figsize=(13, 3.5))
    vmax = np.abs(A).max()
    for ax, mat, title in zip(
        axes_r,
        [A, A_recon, A - A_recon],
        [r"Original $\mathbf{A}$", f"Reconstructed (d={d})", f"Residual (err={error:.0f})"],
    ):
        sns.heatmap(
            mat, ax=ax, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax, cbar=False, xticklabels=False, yticklabels=False
        )
        ax.set_title(title, fontsize=11)
    plt.tight_layout()

    # ---- row 2: error curve ----
    _d_vals = list(range(1, 25))
    _errs = [
        0.5
        * np.linalg.norm(
            A - sum(eigvals_sorted[i] * np.outer(eigvecs_sorted[:, i], eigvecs_sorted[:, i]) for i in range(dv)), "fro"
        )
        ** 2
        for dv in _d_vals
    ]
    fig_e, ax_e = plt.subplots(figsize=(9, 2.5))
    ax_e.plot(_d_vals, _errs, marker="o", ms=4, color="steelblue")
    ax_e.axvline(d, color="crimson", linestyle="--", label=f"d={d}")
    ax_e.set_xlabel("d")
    ax_e.set_ylabel("Frobenius error")
    ax_e.legend()
    ax_e.set_title("Reconstruction error vs d")
    plt.tight_layout()

    mo.vstack([d_slider, fig_r, fig_e])
    return


@app.cell(hide_code=True)
def _(mo):
    _eigenmap = mo.md(r"""
    ## Laplacian Eigenmap — Topology as Geometry

    The **graph Laplacian** is $\mathbf{L} = \mathbf{D} - \mathbf{A}$, where $\mathbf{D}$ is the diagonal degree matrix.
    It has a key identity: for any vector $\mathbf{x} \in \mathbb{R}^n$,

    $$\mathbf{x}^\top \mathbf{L} \mathbf{x} = \sum_{(i,j)\in E}(x_i - x_j)^2$$

    Every edge contributes the *squared difference* of its endpoints' coordinates.
    Minimizing this objective forces connected nodes to land close together in embedding space.

    **The embedding problem:** find $d$-dimensional coordinates $\mathbf{Y} \in \mathbb{R}^{n \times d}$ that respect the topology:

    $$\min_{\mathbf{Y}} \,\mathrm{tr}(\mathbf{Y}^\top \mathbf{L} \mathbf{Y}), \quad \text{subject to} \quad \mathbf{Y}^\top \mathbf{D} \mathbf{Y} = \mathbf{I}$$

    The constraint $\mathbf{Y}^\top \mathbf{D} \mathbf{Y} = \mathbf{I}$ prevents the trivial solution $\mathbf{Y}=\mathbf{0}$ and removes arbitrary scaling.

    **Solution:** the eigenvectors of $\mathbf{L}$ with the $d$ *smallest non-zero* eigenvalues.
    We skip $\mathbf{x}_1 = \mathbf{1}/\sqrt{N}$ (eigenvalue 0 — the trivial constant embedding).
    The next eigenvectors, starting from the **Fiedler vector** $\mathbf{x}_2$, capture the most informative structure.

    The interactive plot below confirms this directly: nodes that are close in the *graph* end up close in the *plane*.
    """)
    mo.vstack([_eigenmap])
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
    _n2v_header = mo.md(r"""
    ## node2vec — Neural Embeddings

    **word2vec (skip-gram):** given a word $w_t$, predict its neighbors within window $c$:
    $$\mathcal{L}=\sum_t\sum_{0<|j|\le c}\log P(w_{t+j}\mid w_t), \qquad P(w_O\mid w_I)=\frac{\exp(\mathbf{v}_{w_O}^\top\mathbf{v}_{w_I})}{\sum_w\exp(\mathbf{v}_w^\top\mathbf{v}_{w_I})}$$

    Each word gets an embedding $\mathbf{v}_w\in\mathbb{R}^d$. Co-occurring words end up close in embedding space.
    **node2vec:** replace words with nodes and sentences with random walks, then apply skip-gram.
    """)

    _walk_col = mo.md(r"""
    **Biased transition probability** (walk is at $v$, came from $t$):
    $$P(\text{next}=x\mid v,t)\;\propto\;\begin{cases}1/p & d(t,x)=0 \;\text{(backtrack to } t\text{)}\\1 & d(t,x)=1 \;\text{(move to neighbor of } t\text{)}\\1/q & d(t,x)=2 \;\text{(explore further)}\end{cases}$$
    """)

    _pq_col = mo.md(r"""
    **Interpreting $p$ and $q$:**

    $p$ controls **return**. Low $p$ ($<1$) makes the walk backtrack to $t$ more often, sampling the *local* neighborhood repeatedly (BFS-like).

    $q$ controls **explore**. Low $q$ ($<1$) pushes the walk away from $t$, discovering *distant* parts of the graph (DFS-like).

    Together they let you trade off between **community membership** (low $q$, homophily) and **structural role** (low $p$, structural equivalence).
    """)

    mo.vstack(
        [
            _n2v_header,
            mo.hstack([_walk_col, _pq_col], gap=40),
        ]
    )
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
def _(mo):
    _fig_caption = mo.md(r"""
    ### node2vec in Action: Homophily vs. Structural Equivalence

    Depending on $p$ and $q$, node2vec discovers fundamentally different patterns.
    With low $q$ (DFS-like walks), nodes in the *same community* land close together.
    With low $p$ (BFS-like walks), nodes with the *same structural role* land close — even if they belong to different communities.
    The figures below, from the original paper, show this on the Les Misérables character network.
    """)

    _walk_img = mo.vstack(
        [
            mo.Html("""<img src="https://snap.stanford.edu/node2vec/walk.png"
             style="max-width:100%; border-radius:6px;" />"""),
            mo.md("*Figure 1 from Grover & Leskovec (2016): biased walk transitions controlled by p and q.*"),
        ]
    )

    _embed_img = mo.vstack(
        [
            mo.Html("""<img src="https://snap.stanford.edu/node2vec/homo.png"
             style="max-width:100%; border-radius:6px;" />"""),
            mo.md("*Figure 3: node2vec embedding on Les Misérables network. Colors = community membership.*"),
        ]
    )

    mo.vstack(
        [
            _fig_caption,
            mo.hstack([_walk_img, _embed_img], gap=32),
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
