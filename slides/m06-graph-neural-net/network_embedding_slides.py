import marimo

__generated_with = "0.23.2"
app = marimo.App(
    width="full",
    layout_file="layouts/network_embedding_slides.slides.json",
    css_file="marimo_lecture_note_theme.css",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import igraph as ig
    import networkx as nx
    from pyvis.network import Network
    import pandas as _pd
    import altair as _alt

    _alt.data_transformers.disable_max_rows()
    return ig, mo, np, nx


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell
def _(np, nx):
    G_nx = nx.karate_club_graph()
    _club = nx.get_node_attributes(G_nx, "club")
    node_order = sorted(G_nx.nodes(), key=lambda v: (0 if _club[v] == "Mr. Hi" else 1, v))
    A = nx.to_numpy_array(G_nx, nodelist=node_order, dtype=float)
    club_sign = np.array([1.0 if _club[v] == "Mr. Hi" else -1.0 for v in node_order], dtype=float)
    return (A,)


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _(mo, n):
    r_outer_slider = mo.ui.slider(1, max(1, n), value=min(4, n), label="r")
    rgse_block_slider = mo.ui.slider(4, max(4, min(12, n)), value=min(8, n), label="block size")
    rgse_terms_slider = mo.ui.slider(1, 4, value=4, label="shown terms")
    rgse_quantile_slider = mo.ui.slider(0.80, 0.99, value=0.95, step=0.01, label="contrast")

    mo.hstack([r_outer_slider, rgse_block_slider, rgse_terms_slider, rgse_quantile_slider], gap=8)
    return (
        r_outer_slider,
        rgse_block_slider,
        rgse_quantile_slider,
        rgse_terms_slider,
    )


@app.cell(hide_code=True)
def _(
    A,
    mo,
    np,
    r_outer_slider,
    rgse_block_slider,
    rgse_quantile_slider,
    rgse_terms_slider,
):
    import pandas as _pd
    import altair as _alt

    _n_show = int(rgse_block_slider.value)
    _r = int(r_outer_slider.value)
    _n_terms_cap = int(rgse_terms_slider.value)
    _q = float(rgse_quantile_slider.value)

    _A_sub = A[:_n_show, :_n_show]
    _w, _V = np.linalg.eigh(_A_sub)
    _ord = np.argsort(np.abs(_w))[::-1]
    _w, _V = _w[_ord], _V[:, _ord]

    _terms_all = [_w[_i] * np.outer(_V[:, _i], _V[:, _i]) for _i in range(_n_show)]
    _sum_r = sum(_terms_all[:_r])
    _n_terms = max(1, min(_n_terms_cap, _r, _n_show))
    _terms_disp = _terms_all[:_n_terms]

    _term_scales = []
    for _t in _terms_disp:
        _s = float(np.quantile(np.abs(_t), _q))
        _term_scales.append(max(_s, 1e-9))
    _vsum = max(float(np.quantile(np.abs(_sum_r), _q)), 1e-9)


    def _contrast(_M, _v):
        _x = np.clip(_M / _v, -1.0, 1.0)
        return np.sign(_x) * np.sqrt(np.abs(_x))


    def _mat_df(_M, _v):
        _n = _M.shape[0]
        _Mc = _contrast(_M, _v)
        return _pd.DataFrame(
            [
                {"row": i, "col": j, "value": float(_M[i, j]), "value_c": float(_Mc[i, j])}
                for i in range(_n)
                for j in range(_n)
            ]
        )


    def _heat(_M, _title, _v):
        _df = _mat_df(_M, _v)
        return (
            _alt.Chart(_df)
            .mark_rect()
            .encode(
                x=_alt.X("col:O", axis=None),
                y=_alt.Y("row:O", axis=None),
                color=_alt.Color("value_c:Q", scale=_alt.Scale(domain=[-1, 1], scheme="redblue"), legend=None),
                tooltip=["row:O", "col:O", _alt.Tooltip("value:Q", format=".3f")],
            )
            .properties(width=150, height=150, title=_title)
        )


    def _sym(_txt):
        return (
            _alt.Chart(_pd.DataFrame({"t": [_txt]}))
            .mark_text(fontSize=28, fontWeight="bold")
            .encode(text="t:N")
            .properties(width=24, height=150)
        )


    parts = []
    for _i in range(_n_terms):
        _ii = _i + 1
        parts.append(_heat(_terms_disp[_i], f"lambda{_ii} q{_ii} q{_ii}^T", _term_scales[_i]))
        if _i < _n_terms - 1:
            parts.append(_sym("+"))
    parts.extend([_sym("="), _heat(_sum_r, f"partial sum (r={_r})", _vsum)])

    _rgse_fig = _alt.hconcat(*parts, spacing=4).resolve_scale(color="independent").configure_axis(grid=False)
    _fro = float(np.linalg.norm(_A_sub - _sum_r, "fro"))

    mo.vstack(
        [
            mo.md(f"reconstruction gap: ||A - sum||_F = {_fro:.2f}"),
            _rgse_fig,
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo, n):
    rw_start = mo.ui.slider(0, n - 1, value=0, label="Start node")
    rw_steps = mo.ui.slider(2, 35, value=12, label="Steps")
    rw_p = mo.ui.slider(0.25, 4.0, value=1.0, step=0.25, label="p")
    rw_q = mo.ui.slider(0.25, 4.0, value=1.0, step=0.25, label="q")
    return rw_p, rw_q, rw_start, rw_steps


@app.cell(hide_code=True)
def _(
    G,
    biased_rw,
    graph_layout_xy,
    mo,
    n,
    np,
    rw_p,
    rw_q,
    rw_start,
    rw_steps,
):
    import pandas as _pd
    import altair as _alt

    _walk = biased_rw(G, rw_start.value, rw_steps.value, rw_p.value, rw_q.value, seed=0)
    _wset = set(_walk)

    rw_coords = graph_layout_xy

    _bg_edges = _pd.DataFrame(
        [
            {"x": rw_coords[e[0], 0], "y": rw_coords[e[0], 1], "x2": rw_coords[e[1], 0], "y2": rw_coords[e[1], 1]}
            for e in G.get_edgelist()
        ]
    )

    _path_df = _pd.DataFrame(
        [
            {
                "step": i,
                "x": rw_coords[_walk[i], 0],
                "y": rw_coords[_walk[i], 1],
                "x2": rw_coords[_walk[i + 1], 0],
                "y2": rw_coords[_walk[i + 1], 1],
            }
            for i in range(len(_walk) - 1)
        ]
    )

    _node_df = _pd.DataFrame(
        {
            "node": np.arange(n).astype(str),
            "x": rw_coords[:, 0],
            "y": rw_coords[:, 1],
            "visited": [str(v in _wset) for v in range(n)],
        }
    )

    _start_df = _pd.DataFrame({"x": [rw_coords[_walk[0], 0]], "y": [rw_coords[_walk[0], 1]]})
    _end_df = _pd.DataFrame({"x": [rw_coords[_walk[-1], 0]], "y": [rw_coords[_walk[-1], 1]]})

    base_edges = (
        _alt.Chart(_bg_edges).mark_rule(color="#e5e7eb", opacity=0.95).encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )
    path_edges = (
        _alt.Chart(_path_df)
        .mark_rule(strokeWidth=2.4)
        .encode(
            x="x:Q",
            y="y:Q",
            x2="x2:Q",
            y2="y2:Q",
            color=_alt.Color("step:Q", scale=_alt.Scale(scheme="blues"), legend=None),
            tooltip=["step:Q"],
        )
    )

    nodes = (
        _alt.Chart(_node_df)
        .mark_circle(size=150, strokeWidth=1.0)
        .encode(
            x="x:Q",
            y="y:Q",
            color=_alt.Color(
                "visited:N", scale=_alt.Scale(domain=["False", "True"], range=["#e5e7eb", "#60a5fa"]), legend=None
            ),
            stroke=_alt.Stroke(
                "visited:N", scale=_alt.Scale(domain=["False", "True"], range=["#9ca3af", "#1d4ed8"]), legend=None
            ),
        )
    )

    start_mark = (
        _alt.Chart(_start_df)
        .mark_point(size=320, shape="triangle-up", color="#f97316", stroke="white", strokeWidth=1.1)
        .encode(x="x:Q", y="y:Q")
    )
    end_mark = (
        _alt.Chart(_end_df)
        .mark_point(size=230, shape="diamond", color="#a855f7", stroke="white", strokeWidth=1.0)
        .encode(x="x:Q", y="y:Q")
    )
    _rw_label_layer = _alt.Chart(_node_df).mark_text(fontSize=9, fontWeight="bold").encode(x="x:Q", y="y:Q", text="node:N")

    fig_rw = (
        _alt.layer(base_edges, path_edges, nodes, start_mark, end_mark, _rw_label_layer)
        .properties(width=620, height=420, title=f"Walk  p={rw_p.value}  q={rw_q.value}  ({rw_steps.value} steps)")
        .configure_axis(grid=False, labels=False, ticks=False, domain=False)
    )

    _walk_str = " -> ".join(str(_wv) for _wv in _walk)
    _controls = mo.vstack(
        [
            mo.hstack([rw_start, rw_steps, rw_p, rw_q], gap=8),
            mo.md(f"**Walk:** `{_walk_str}`"),
        ],
        gap=4,
    )

    mo.vstack([_controls, fig_rw], gap=4)
    return


@app.cell(hide_code=True)
def _(mo, n):
    k_slider = mo.ui.slider(2, max(2, min(6, n - 1)), value=2, label="k (clusters)")
    return (k_slider,)


@app.cell(hide_code=True)
def _(A, G, graph_layout_xy, k_slider, mo, n, np):
    import pandas as _pd
    import altair as _alt

    k = k_slider.value

    D_mat = np.diag(A.sum(axis=1))
    L = D_mat - A
    eigvals_L, eigvecs_L = np.linalg.eigh(L)
    fiedler_vecs = eigvecs_L[:, 1:k]

    from sklearn.cluster import KMeans

    cluster_labels = (
        (fiedler_vecs[:, 0] > 0).astype(int)
        if k == 2
        else KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(fiedler_vecs)
    )

    coords = graph_layout_xy

    _edge_df = _pd.DataFrame(
        [
            {"x": coords[e[0], 0], "y": coords[e[0], 1], "x2": coords[e[1], 0], "y2": coords[e[1], 1]}
            for e in G.get_edgelist()
        ]
    )
    _node_df = _pd.DataFrame(
        {
            "node": np.arange(n).astype(str),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": cluster_labels.astype(int),
        }
    )

    _g_edges = _alt.Chart(_edge_df).mark_rule(color="#9ca3af", opacity=0.35).encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    _g_nodes = (
        _alt.Chart(_node_df)
        .mark_circle(size=120, stroke="white", strokeWidth=0.9)
        .encode(
            x="x:Q",
            y="y:Q",
            color=_alt.Color("cluster:N", scale=_alt.Scale(scheme="tableau10"), legend=None),
            tooltip=["node:N", "cluster:N"],
        )
    )
    _g_text = _alt.Chart(_node_df).mark_text(fontSize=8, dy=-8).encode(x="x:Q", y="y:Q", text="node:N")
    chart_graph = _alt.layer(_g_edges, _g_nodes, _g_text).properties(
        width=350, height=260, title=f"Spectral clustering  k={k}"
    )

    _emb_df = _pd.DataFrame(
        {
            "node": np.arange(n).astype(str),
            "x2": eigvecs_L[:, 1],
            "x3": eigvecs_L[:, 2] if n > 2 else eigvecs_L[:, 1],
            "cluster": cluster_labels.astype(int),
        }
    )

    _emb_pts = (
        _alt.Chart(_emb_df)
        .mark_circle(size=80)
        .encode(
            x=_alt.X("x2:Q", title="x2 (Fiedler vector)"),
            y=_alt.Y("x3:Q", title="x3"),
            color=_alt.Color("cluster:N", scale=_alt.Scale(scheme="tableau10"), legend=None),
            tooltip=["node:N", "cluster:N", _alt.Tooltip("x2:Q", format=".3f"), _alt.Tooltip("x3:Q", format=".3f")],
        )
    )
    _emb_text = _alt.Chart(_emb_df).mark_text(fontSize=8, dy=-8).encode(x="x2:Q", y="x3:Q", text="node:N")
    chart_em = _alt.layer(_emb_pts, _emb_text).properties(width=350, height=260, title="Laplacian eigenmap")

    _fig_clust = _alt.hconcat(chart_graph, chart_em, spacing=10).configure_axis(grid=False)
    mo.vstack([k_slider, _fig_clust], gap=4)
    return


@app.cell(hide_code=True)
def _(A, ig, np):
    # Build graph layout from A and expose n for downstream controls.
    n = A.shape[0]
    G = ig.Graph.Adjacency((A > 0).tolist(), mode="undirected")
    graph_layout_xy = np.array(G.layout_kamada_kawai().coords)
    return G, graph_layout_xy, n


@app.cell(hide_code=True)
def _(A, n, np):
    rng = np.random.default_rng(42)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] > 0]
    rng.shuffle(edges)
    n_test = max(2, min(len(edges), len(edges) // 5 if len(edges) > 0 else 2))
    test_pos = edges[:n_test]

    A_train = A.copy()
    for i, j in test_pos:
        A_train[i, j] = A_train[j, i] = 0

    non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 0]
    rng.shuffle(non_edges)
    test_neg = non_edges[:n_test]
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
    _placeholder_ecfg = None
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
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

    return (biased_rw,)


@app.cell(hide_code=True)
def _():
    _ = None
    return


@app.cell(hide_code=True)
def _():
    _ = None
    return


if __name__ == "__main__":
    app.run()
