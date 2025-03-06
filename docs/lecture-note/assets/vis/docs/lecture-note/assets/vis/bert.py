import marimo as mo

app = mo.App()


@app.cell
def _():
    import marimo as mo

    return mo


@app.cell
def _(mo):
    title = mo.md("# **BERT**")
    subtitle = mo.md("### *Bidirectional Encoder Representations from Transformers*")
    author = mo.md("#### Sadamori Kojaku")
    image = mo.md(
        "![BERT MLM](https://cdn.botpenguin.com/assets/website/BERT_c35709b509.webp)"
    )

    _text = mo.vstack(
        [mo.center(title), mo.center(subtitle), mo.md("<br>"), mo.center(author)],
        align="center",
    )
    mo.hstack([_text, image], justify="center")
    return


if __name__ == "__main__":
    app.run()
