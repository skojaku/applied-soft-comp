# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "markdown==3.7",
#     "matplotlib==3.10.1",
#     "numpy==2.2.3",
#     "pandas==2.2.3",
#     "scikit-learn==1.6.1",
#     "torch==2.6.0",
#     "tqdm==4.67.1",
#     "transformers==4.49.0",
# ]
# ///
import marimo

__generated_with = "0.11.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        <div style="display: flex; align-items: center;">
            <div style="flex: 1;">
                <span style="font-size: 1.5em;">Bidirectional Encoder Representations from Transformers (BERT)</span>
                <p>Sadamori Kojaku</p>
            </div>
            <div style="flex: 1;">
                <img src="https://cdn.botpenguin.com/assets/website/BERT_c35709b509.webp" width="400">
            </div>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # BERT: A successor of ELMo

        ELMo addressed polysemy with **bidirectional LSTMs**.


        ![](https://miro.medium.com/v2/resize:fit:1400/1*rawUFwyB0hRnkNuKw3c-tA.png)

        BERT advances this by using **transformers** and a **self-attention mechanism**, significantly improving context understanding.

        ![](https://trituenhantao.io/wp-content/uploads/2019/10/bert-finetunning.png)

        Due to its powerful capabilities in tasks like **question answering** and **text classification**, BERT has become foundational in NLP, even enhancing Google's search engine.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Architecture of BERT

        **BERT** consists of stacked 12 **encoder transformer layers**. No decoder is used.

        Each layer progressively refines **token embeddings**, making them increasingly context-aware and effective for NLP tasks.

        ![](https://www.researchgate.net/publication/372906672/figure/fig2/AS:11431281179224913@1691164535766/BERT-model-architecture.ppm)

        # 🧠 Pre-training BERT

        - BERT is pre-trained on massive text datasets like Wikipedia and BooksCorpus.
        - During this phase, BERT learns language patterns, context, and semantic relationships without human supervision.
        - This foundational knowledge enables BERT to generalize effectively across various NLP tasks. 🔄

        BERT is pre-trained on two objectives:

        - **Masked Language Modeling (MLM)**
        - **Next Sentence Prediction (NSP)**

        Let's see how these objectives work.

        ## Masked Language Modeling (MLM)


        <div style="display: flex; justify-content: center;">
        <svg xmlns="http://www.w3.org/2000/svg"  width="600" viewBox="0 0 800 500">

          <!-- Title -->
          <text x="400" y="40" font-family="Arial" font-size="22" font-weight="bold" text-anchor="middle" fill="#333">BERT Masking Process for MLM Training</text>

          <!-- Original sentence -->
          <rect x="50" y="80" width="700" height="60" rx="5" ry="5" fill="#e3f2fd" stroke="#2196f3" stroke-width="2" />
          <text x="400" y="100" font-family="Arial" font-size="18" font-weight="bold" text-anchor="middle">Original Sentence</text>
          <g transform="translate(80, 0)">
            <g font-family="monospace" font-size="16" text-anchor="middle">
              <text x="100" y="130" fill="#333">"the"</text>
              <text x="170" y="130" fill="#333">"cat"</text>
              <text x="240" y="130" fill="#333">"sat"</text>
              <text x="310" y="130" fill="#333">"on"</text>
              <text x="380" y="130" fill="#333">"the"</text>
              <text x="450" y="130" fill="#333">"mat"</text>
              <text x="520" y="130" fill="#333">"."</text>
            </g>
          </g>

          <!-- Random 15% Selection -->
          <rect x="50" y="160" width="700" height="60" rx="5" ry="5" fill="#e8f5e9" stroke="#4caf50" stroke-width="2" />
          <text x="400" y="180" font-family="Arial" font-size="18" font-weight="bold" text-anchor="middle">Select 15% of Tokens Randomly</text>
          <g transform="translate(80, 0)">
          <g font-family="monospace" font-size="16" text-anchor="middle">
            <text x="100" y="210" fill="#333">"the"</text>
            <text x="170" y="210" fill="#333">"cat"</text>
            <rect x="225" y="195" width="30" height="20" fill="#ffeb3b" rx="3" ry="3" />
            <text x="240" y="210" fill="#333">"sat"</text>
            <text x="310" y="210" fill="#333">"on"</text>
            <text x="380" y="210" fill="#333">"the"</text>
            <text x="450" y="210" fill="#333">"mat"</text>
            <text x="520" y="210" fill="#333">"."</text>
          </g>
          </g>

          <!-- Three masking options -->
          <g>
            <!-- 80% MASK -->
            <rect x="50" y="240" width="220" height="180" rx="5" ry="5" fill="#e0f7fa" stroke="#00bcd4" stroke-width="2" />
            <text x="160" y="270" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">80% of time:</text>
            <text x="160" y="290" font-family="Arial" font-size="16" text-anchor="middle">Replace with [MASK]</text>
            <g font-family="monospace" font-size="14" text-anchor="middle">
              <text x="90" y="335" fill="#333">... 'cat'</text>
              <text x="160" y="335" fill="#333" font-weight="bold">[MASK]</text>
              <text x="240" y="335" fill="#333">'on' ...</text>
            </g>
            <text x="160" y="370" font-family="Arial" font-size="13" fill="#444" text-anchor="middle">Prevents model from</text>
            <text x="160" y="385" font-family="Arial" font-size="13" fill="#444" text-anchor="middle">relying solely on</text>
            <text x="160" y="400" font-family="Arial" font-size="13" fill="#444" text-anchor="middle">the mask token</text>
          </g>

          <g>
            <!-- 10% Random -->
            <rect x="290" y="240" width="220" height="180" rx="5" ry="5" fill="#fff3e0" stroke="#ff9800" stroke-width="2" />
            <text x="400" y="270" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">10% of time:</text>
            <text x="400" y="290" font-family="Arial" font-size="16" text-anchor="middle">Replace with random word</text>
            <g font-family="monospace" font-size="14" text-anchor="middle">
              <text x="340" y="335" fill="#333">... 'cat'</text>
              <text x="400" y="335" fill="#333" font-weight="bold" fill="#d32f2f">'dog'</text>
              <text x="460" y="335" fill="#333">'on' ...</text>
            </g>
            <text x="400" y="370" font-family="Arial" font-size="13" fill="#444" text-anchor="middle">Adapts model to handle</text>
            <text x="400" y="385" font-family="Arial" font-size="13" fill="#444" text-anchor="middle">noise and incorrect</text>
            <text x="400" y="400" font-family="Arial" font-size="13" fill="#444" text-anchor="middle">predictions</text>
          </g>

          <g>
            <!-- 10% Same -->
            <rect x="530" y="240" width="220" height="180" rx="5" ry="5" fill="#f3e5f5" stroke="#9c27b0" stroke-width="2" />
            <text x="640" y="270" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">10% of time:</text>
            <text x="640" y="290" font-family="Arial" font-size="16" text-anchor="middle">Keep unchanged</text>
            <g font-family="monospace" font-size="16" text-anchor="middle">
              <text x="580" y="335" fill="#333">... 'cat'</text>
              <text x="640" y="335" fill="#333" font-weight="bold">'sat'</text>
              <text x="700" y="335" fill="#333">'on' ...</text>
            </g>
            <text x="640" y="370" font-family="Arial" font-size="13" fill="#444" text-anchor="middle">Forces model to make</text>
            <text x="640" y="385" font-family="Arial" font-size="13" fill="#444" text-anchor="middle">predictions for all</text>
            <text x="640" y="400" font-family="Arial" font-size="13" fill="#444" text-anchor="middle">selected positions</text>
          </g>

          <!-- Prediction task -->
          <rect x="150" y="440" width="500" height="40" rx="5" ry="5" fill="#ede7f6" stroke="#673ab7" stroke-width="2" />
          <text x="400" y="465" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">Prediction Task: Model must predict "sat" at masked position</text>

          <!-- Arrows -->
          <g stroke="#666" stroke-width="2" fill="none">
            <path d="M400,145 L400,160" marker-end="url(#arrowhead)" />
            <path d="M160,225 L160,240" marker-end="url(#arrowhead)" />
            <path d="M400,225 L400,240" marker-end="url(#arrowhead)" />
            <path d="M640,225 L640,240" marker-end="url(#arrowhead)" />
            <path d="M160,420 L280,440" marker-end="url(#arrowhead)" />
            <path d="M400,420 L400,440" marker-end="url(#arrowhead)" />
            <path d="M640,420 L520,440" marker-end="url(#arrowhead)" />
          </g>

          <!-- Arrowhead marker -->
          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
            </marker>
          </defs>
        </svg>
        </div>

        - In MLM, some tokens are masked, and the model must predict the masked tokens.
        - Three ways to mask the tokens:
            - Replace the token with `[MASK]`
            - Replace the token with a random word
            - Keep the token unchanged

        - Given a sentence, the model must recover the original tokens based on the context.


        ## Next Sentence Prediction (NSP)

        ![](https://amitness.com/posts/images/bert-nsp.png)

        - NSP is another objective for predicting whether two sentences are consecutive or not.
        - This equips BERT with the ability to understand the relationships between sentences.
        - Two special tokens are introduced to structure the sentence pairs:
            - **[CLS] token** at the start of the first sentence (classification token)
            - **[SEP] token** at the end of the sentences (separator token)

        - Two sentences are then concatenated into one sequence, for example, as follows:

        $$
        \\text{``[CLS] }\\underbrace{\\text{I went to the store}}_{\\text{Sentence 1}}\\text{ [SEP] }\\underbrace{\\text{They were out of milk}}_{\\text{Sentence 2}}\\text{ [SEP]}".
        $$

        - The text is then fed into the BERT model, producing the embeddings of the tokens.
        - The embedding of the [CLS] token is then used to predict whether the two sentences are consecutive or not.

        > *Note*: To inform BERT which tokens belong to which sentence, BERT additionally uses so-called `segment embeddings`. But recent transformer architectures (e.g., GPT models, RoBERTa, LLaMA) typically omit explicit segment embeddings, opting instead for simplified positional encodings or relying purely on positional embeddings.
        > <center><img src="https://towardsdatascience.com/wp-content/uploads/2024/05/1w4r6Hz2IF-Uvo4YHg2-bCw-1.png" width="500"></center>

        # Let's play with BERT 🎁

        The best way to understand BERT is to play with it 😉.

        We will use the [transformers](https://huggingface.co/docs/transformers/index) library to load the model and the tokenizer.
        """
    )
    return


@app.cell
def _():
    # Import transformers
    from transformers import AutoTokenizer, AutoModel

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    # Load the model
    model = AutoModel.from_pretrained(
        "bert-base-uncased"
    )

    # Set the model to evaluation mode
    model = model.eval()
    return AutoModel, AutoTokenizer, model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        - `transformers` library provides easy access to NLP models
        - `AutoTokenizer` and `AutoModel` automatically load the right tokenizer and model
        - `model.eval()` prepares the model for inference
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ##Tokenizer

        The tokenizer breaks text into tokens that BERT can understand.
        Let's see what the tokenizer does with the following text.

        ```python
        >>> text = "Binghamton University"
        ```

        We will then break the text into the tokens, and then convert the tokens into token IDs (numbers) that BERT can understand.

        We will add special tokens to the text to help BERT understand the structure of the text.
        """
    )
    return


@app.cell
def _(tokenizer):
    # The text to tokenize
    text = "Binghamton University"

    # Tokenize the text. We add special tokens to the text.
    tokens = tokenizer.tokenize(text, add_special_tokens=True)

    # Convert the tokens into token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f"Text: '{text}'")
    print(f"Tokenized: {tokens}")
    print(f"Token IDs: {token_ids}")
    return text, token_ids, tokens


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Embedding tokens
        Now the preparation is done. Let us feed the token IDs to the model and get the embeddings of the tokens.
        """
    )
    return


@app.cell
def _(model, token_ids):
    import torch

    # Convert the token IDs (numpy array) into a torch tensor
    token_ids_tensor = torch.tensor([token_ids])

    # Feed the token IDs to the model
    outputs = model(token_ids_tensor, output_hidden_states=True)
    return outputs, token_ids_tensor, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        The `outputs` contains:

        1. **Input token embeddings** before the first transformer module
        2. **Output token embeddings** after each transformer module
        3. **Attention scores** of the tokens

        Let's retrieve the token embeddings (1 and 2) by

        ```python
        >>> outputs.hidden_states
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(outputs):
    outputs.hidden_states
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        - There are 13 tensors in `outputs.hidden_states` (1 input token embedding before transformers + 12 transformer outputs).
        - `outputs.hidden_states[i]` is the token embeddings before the i-th transformer module.
        - Each tensor has the shape `(batch_size, sequence_length, hidden_size)`.
            - `batch_size` is the number of sentences in the batch.
            - `sequence_length` is the number of tokens in the sentence.
            - `hidden_size` is the size of the hidden state of the transformer.
        """
    )
    return


@app.cell
def _(outputs):
    # Get the last hidden state
    last_hidden_state = outputs.hidden_states[-1]

    # Get the shape of the last hidden state
    shape = last_hidden_state.shape
    print(shape)
    return last_hidden_state, shape


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Each token embedding can be retrieved by:""")
    return


@app.cell
def _(last_hidden_state):
    # The position of the token in the sentence
    token_position = 3

    # Get the embedding of the token
    token_embedding = last_hidden_state[0, token_position, :]

    print(token_embedding[:10]) # Truncated for brevity
    return token_embedding, token_position


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Batch processing

        - It is often the case that we have multiple sentences to process.
        - Processing them one by one is not efficient (overhead of tokenization, model loading, etc.)
        - We can process them in a batch and get the embeddings for all sentences at once.

        Sounds straightforward! But there is a challenge 🤔, i.e., sentences have different lengths, so that we may not be able to pack them into a single tensor.

        Let us consider the following two text:

        ```python
        >>> text1 = "Binghamton University"
        >>> text2 = "State University of New York"
        ```

        As in the previous example, let's tokenize the sentences and convert them into token IDs.
        """
    )
    return


@app.cell
def _(tokenizer):
    text1 = "Binghamton University"
    text2 = "State University of New York"

    # Tokenize the text. We add special tokens to the text.
    tokens1 = tokenizer.tokenize(text1, add_special_tokens=True)
    tokens2 = tokenizer.tokenize(text2, add_special_tokens=True)

    token_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
    token_ids2 = tokenizer.convert_tokens_to_ids(tokens2)

    print(f"Token IDs of text1: {token_ids1}")
    print(f"Token IDs of text2: {token_ids2}")
    return text1, text2, token_ids1, token_ids2, tokens1, tokens2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        To feed them into the model in batch, we need to pack them into a single token id tensor as follows:

        ```python
        >>> token_ids = torch.tensor([token_ids1, token_ids2])
        ```

        which won't work because the sentences have different lengths!

        To address this issue of inconsistent sequence lengths, we can pad the sentences with special tokens (e.g., `[PAD]`) so that they have the same length.

        The attention for the padded tokens should be masked out so that padding does not change the result.

        ![](https://miro.medium.com/v2/resize:fit:1216/1*N6p-Oknq_QXV0vSX5KvwdQ.png)

        In `transformers`, we can use `tokenizer` to generate the padded tensor and the attention mask.

        > *Note*: We will use `tokenizer` not `tokenizer.tokenize`! The `tokenizer.tokenize` only breaks text into tokens, while `tokenizer` can generate all the input data needed to run BERT.
        """
    )
    return


@app.cell
def _(text1, text2, tokenizer):
    # Tokenize the text. We add special tokens to the text.
    inputs = tokenizer([text1, text2], add_special_tokens=True, padding=True, truncation=True, return_tensors="pt", return_attention_mask = True)

    print(inputs)
    return (inputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Now we can feed the padded sequences and the attention mask into the model. Don't forget to set `output_hidden_states=True` so that we can retrieve the token embeddings.

        Notice that the first dimension is now `2` because we have two sentences.
        """
    )
    return


@app.cell
def _(inputs, model):
    # Process the multiple sentences in batch
    outputs_batch = model(**inputs, output_hidden_states=True)

    last_hidden_state_batch = outputs_batch.hidden_states[-1]

    print(f"Last hidden state batch shape: {last_hidden_state_batch.shape}")
    return last_hidden_state_batch, outputs_batch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Case study: Polysemy resolution

        To illustrate the power of BERT, let us consider a case study of polysemy resolution.

        Consider the following two sentences:

        1. "The bank in the city is closed"

        2. "The bank in the river is closed"

        The word "bank" has two meanings:

        - "bank" as in "financial institution"
        - "bank" as in "side of a river"

        To resolve the polysemy, we can use BERT to get the embeddings of the sentence and the word "bank".
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import sys
    import matplotlib.pyplot as plt
    from tqdm import tqdm as tqdm
    from sklearn.decomposition import PCA
    return PCA, np, pd, plt, sys, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We will use [CoarseWSD-20](https://github.com/danlou/bert-disambiguation/tree/master/data/CoarseWSD-20). The dataset contains sentences with polysemous words and their sense labels. We will see how to use BERT to disambiguate the word senses. Read the [README](https://github.com/danlou/bert-disambiguation/blob/master/data/CoarseWSD-20/README.txt) for more details.""")
    return


@app.cell
def _(pd):
    def load_data(focal_word, is_train, n_samples=100):
        data_type = "train" if is_train else "test"
        data_file = f"https://raw.githubusercontent.com/danlou/bert-disambiguation/master/data/CoarseWSD-20/{focal_word}/{data_type}.data.txt"
        label_file = f"https://raw.githubusercontent.com/danlou/bert-disambiguation/master/data/CoarseWSD-20/{focal_word}/{data_type}.gold.txt"

        data_table = pd.read_csv(
            data_file,
            sep="\t",
            header=None,
            dtype={"word_pos": int, "sentence": str},
            names=["word_pos", "sentence"],
        )
        label_table = pd.read_csv(
            label_file,
            sep="\t",
            header=None,
            dtype={"label": int},
            names=["label"],
        )
        combined_table = pd.concat([data_table, label_table], axis=1)
        return combined_table.sample(n_samples)


    focal_word = "apple"

    train_data = load_data(focal_word, is_train=True)
    return focal_word, load_data, train_data


@app.cell(hide_code=True)
def _(train_data):
    train_data.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The data contains `word_pos`, `label`, and `sentence`. The `word_pos` indicates the token in question for polysemy resolution. The `label` indicates the meaning of the token in the sentence.

        Now, let's embed the tokens in the sentences in batch.

        Since we have multiple layers in the BERT model, we have freedom to choose which layer to use for the embeddings. Choose the layer for the embedding, and see how the embedding changes.
        """
    )
    return


@app.cell
def _(model, tokenizer, torch, train_data):
    from collections import defaultdict

    # Batch size
    batch_size = 128

    all_labels = []
    all_sentences = []
    all_embeddings = defaultdict(list)

    # Process data in batches
    for i in range(0, len(train_data), batch_size):
        batch = train_data.iloc[i:i+batch_size]

        # Prepare batch data
        batch_sentences = batch["sentence"].tolist()
        batch_focal_indices = batch["word_pos"].tolist()
        batch_labels = batch["label"].tolist()

        # Tokenize all sentences in the batch
        encoded_inputs = tokenizer(
            batch_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )

        # Get BERT embeddings for the batch
        _outputs_batch = model(**encoded_inputs, output_hidden_states=True)

        # Get the focal token embeddings
        for layer_id in range(len(_outputs_batch.hidden_states)):
            focal_token_embeddings = [
                _outputs_batch.hidden_states[layer_id][i, batch_focal_indices, :]
                for i, batch_focal_indices in enumerate(batch_focal_indices)
            ]
            all_embeddings[layer_id]+=focal_token_embeddings


        # Stack the embeddings and labels
        all_labels = all_labels + batch_labels
        all_sentences = all_sentences + batch_sentences

    for layer_id in all_embeddings.keys():
        all_embeddings[layer_id] = torch.vstack(all_embeddings[layer_id]).detach().numpy()
    return (
        all_embeddings,
        all_labels,
        all_sentences,
        batch,
        batch_focal_indices,
        batch_labels,
        batch_sentences,
        batch_size,
        defaultdict,
        encoded_inputs,
        focal_token_embeddings,
        i,
        layer_id,
    )


@app.cell(hide_code=True)
def _(mo, slider_focal_layer_ids):
    _text = mo.md("Now, let's visualize the embeddings in 2D using PCA.")
    mo.vstack([_text, slider_focal_layer_ids])
    return


@app.cell(hide_code=True)
def _(
    PCA,
    all_embeddings,
    all_labels,
    all_sentences,
    pd,
    slider_focal_layer_ids,
):
    import altair as alt
    # Apply PCA to reduce dimensions to 2D
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(all_embeddings[slider_focal_layer_ids.value])

    # Create a DataFrame for Altair
    df_chart = pd.DataFrame({
        'x': xy[:, 0],
        'y': xy[:, 1],
        'label': all_labels,
        'sentence': all_sentences
    })

    chart = alt.Chart(df_chart).mark_circle(size=120).encode(
        x=alt.X('x:Q', title='PCA 1'),
        y=alt.Y('y:Q', title='PCA 2'),
        color=alt.Color('label:N', legend=alt.Legend(title="Label")),
        tooltip=['label', 'sentence']
    ).properties(
        width=700,
        height=500,
        title='Word Embeddings Visualization'
    ).interactive()

    # Display the chart
    chart
    return alt, chart, df_chart, pca, xy


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Case Study 2: Implicit associations in language

        Human language contains implicit associations between concepts that reflect cultural norms, stereotypes, and common patterns. These associations can be captured by language models like BERT, which learn statistical patterns from large text corpora.

        Let's explore how we can use BERT to capture implicit associations between concepts, using the example of noun-color associations.

        We set up a MLM task using the following template:

        "When asked about its color, {object} is described as [MASK]."

        - The {object} is any noun, e.g., "grass", "sky", "banana", "blood", "snow", etc.
        - BERT will predict the masked tokens in the sentence.
        - Since there is no words associated explicitly with the color to be predicted, the prediction is expected to reflect the BERT's understanding of the semantic meaning of the {object}.

        ### Implementation



        To run this task, we will use `BertForMaskedLM` an adapted version of `AutoModel` for masked language modeling.

        - **`BertForMaskedLM`** is specifically designed for BERT models with a language modeling head on top.
        - This specialized model includes a linear layer that projects the hidden states to vocabulary-sized logits, enabling direct prediction of masked tokens.

        <img src="https://www.oreilly.com/api/v2/epubs/9781098136789/files/assets/nlpt_0404.png" style="display: block; margin-left: auto; margin-right: auto;">
        """
    )
    return


@app.cell
def _():
    from transformers import BertForMaskedLM

    model_masked_lm = BertForMaskedLM.from_pretrained('bert-base-uncased')
    return BertForMaskedLM, model_masked_lm


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Now, let's predict the masked word in the sentence.""")
    return


@app.cell
def _(model_masked_lm, tokenizer, torch):
    def predict_masked_word(template, object_name, top_k=5):
        # Fill in the template with the object
        text = template.format(object=object_name)

        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt")

        # Get position of [MASK] token
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

        # Forward pass
        with torch.no_grad():
            outputs = model_masked_lm(**inputs)

        # Get predictions
        logits = outputs.logits  # This is the correct attribute for BertForMaskedLM
        mask_token_logits = logits[0, mask_token_index, :]

        # Get top-k predictions
        top_k_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()

        # Convert token IDs to words
        top_k_words = [tokenizer.convert_ids_to_tokens(token_id) for token_id in top_k_tokens]

        return top_k_words
    return (predict_masked_word,)


@app.cell(hide_code=True)
def _(mo, noun_placeholder, predict_masked_word):
    template  = "When asked about its color, {object} is described as [MASK]."
    top_k = 8

    obj = noun_placeholder.value
    predictions = predict_masked_word(template, obj, top_k)
    results = f"**{obj}**: {', '.join(predictions)}"

    mo.vstack([noun_placeholder,results])
    return obj, predictions, results, template, top_k


@app.cell(hide_code=True)
def _(mo):
    slider_focal_layer_ids = mo.ui.slider(
        0,
        12,
        1,
        4,
        label="Layer to use",
    )
    noun_placeholder = mo.ui.text(
        value="banana",
        label="When asked about its color, {object} is described as [MASK].",
        full_width = True
    )
    return noun_placeholder, slider_focal_layer_ids


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import inspect
    import markdown
    return inspect, markdown, mo


if __name__ == "__main__":
    app.run()
