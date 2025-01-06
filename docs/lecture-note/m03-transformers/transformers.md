---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Transformers

Transformers are the cornerstone of modern NLP that gives rise to the recent success of LLMs. We will learn how transformers work and how they are used to build LLMs.

## A building block of LLMs

Many large language models (LLMs) including GPT-3, GPT-4, and Claude are built based on a stack of *transformer* blocks {footcite:p}`vaswani2017attention`.
Each transformer block takes a sequence of token vectors as input and outputs a sequence of token vectors (sequence-to-sequence!).
Inside each transformer block are essentially three components, i.e., *multi-head attention*, *layer normalization*, and *feed-forward networks*.

```{figure} ../figs/transformer-overview.jpg
:name: transformer-overview
:alt: Transformer Overview
:width: 80%
:align: center

The basic architecture of the transformer-based LLMs.
```

## Attention Mechanism

Perhaps the most crucial component of the transformer is the *attention mechanism*, which allows the model to pay attention to particular parts of the input sequence.


### Self-Attention
In transformers, the attention is called *self-attention*, since the attention is paid within the same sentence, unlike the sequence-to-sequence models that pays attention from one sentence to another. At its core, self-attention is about relationships. When you read the sentence "The cat sat on the mat because it was tired", how do you know what "it" refers to? You naturally look back at the previous words and determine that "it" refers to "the cat". Self-attention works similarly, but does this for every word in relation to every other word, simultaneously.

```{figure} ../figs/transformer-attention.jpg
:name: transformer-attention
:alt: Attention Mechanism
:width: 80%
:align: center

The attention mechanism in transformers.
```

To compute the attention between words, the attention head creates three types of vectors---**query, key, and value**---for each word. Each of these vectors are created by a neural network (w/ single linear layer) that takes the input word as input, and outputs another vector.

```{note}
Think of this like a library system: The Query is what you're looking for, the Keys are like book titles, and the Values are the actual content of the books. When you search (Q) for a specific topic, you match it against book titles (K) to find the relevant content (V).
```

The query and key vectors are used to compute the attention score, which represents how much attention the model pays to each key word for the query word, with a larger score indicating a stronger attention. For example, in the sentence "The cat sat on the mat because it was tired", a good model should pay more attention to "cat" than "mat" for the word "it". The atttention score computed by the dot product of the query and key vectors.  The score is then normalized by the softmax function, with rescaling by $\sqrt{d_k}$ to prevent the score from becoming too large. More formally, the attention score is computed as:

$$
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right),
$$

where $Q \in \mathbb{R}^{n \times d_k}$ is the query matrix containing $n$ query vectors of dimension $d_k$, $K \in \mathbb{R}^{n \times d_k}$ is the key matrix containing $n$ key vectors of dimension $d_k$, and $V \in \mathbb{R}^{n \times d_v}$ is the value matrix containing $n$ value vectors of dimension $d_v$.

The normalized attention score is used as a weight for the weghted sum of the value vectors, which results in *the contextualized vector of the query word*. Putting all the pieces together, the attention mechanism is computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V,
$$

where $V \in \mathbb{R}^{n \times d_v}$ is the value matrix containing $n$ value vectors of dimension $d_v$. In the original paper on transformers {footcite:p}`vaswani2017attention`, the dimension of the query, key, and value vectors are all set to be the same, i.e., $d_k = d_v = d_q = d / h$, where $h$ is the number of attention heads and $d$ is the dimension of the input vector, though this is not a strict requirement.

```{note}
The output of the attention mechanism is the *contextualized vector*, meaning that the vector for a word can vary depending on other words input to the attention module. This is ideal for language modeling, since the meaning of a word can vary depending on the context, e.g., "bank" can mean "river bank" or "financial institution" depending on the words surrounding it.
```

### Multi-Head Attention

Multi-head attention consists of multiple attention heads to enable a model to pay attentions to multiple aspects of the input sequence. Each attention head can have different parameters and thus produces different "contextualized vectors." These different vector are then concatenated and fed into a feed-forward network to produce the final output.

```{figure} ../figs/transformer-multihead-attention.jpg
:name: transformer-multihead-attention
:alt: Multi-Head Attention
:width: 50%
:align: center

Multi-head attention mechanism.
```

## Layer Normalization

```{figure} https://miro.medium.com/v2/resize:fit:1400/0*Agdt1zYwfUxXMJGJ
:name: transformer-layer-normalization
:alt: Layer Normalization
:width: 80%
:align: center

Layer normalization works by normalizing each individual sample across its features. For each sample, it calculates the mean and standard deviation across all feature dimensions, then uses these statistics to normalize that sample's values.
```

*Layer normalization* is a technique used to stabilize the training of deep neural networks. It mitigates the problem of too large or too small input values, which can cause the network to become unstable. This normalization shifts and scales the input values to prevent this issue. More specifically, the layer normalization is computed as:

$$
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sigma} + \beta,
$$

where $\mu$ and $\sigma$ are the mean and standard deviation of the input, $\gamma$ is the scaling factor, and $\beta$ is the shifting factor. The variables $\gamma$ and $\beta$ are learnable parameters that are initialized to 1 and 0, respectively, and are updated during training.


## Wiring it all together

### Structure of a Transformer Block

Now, we have all the components to build a transformer block. Let's wire them together.

```{figure} ../figs/transformer-wired-components.jpg
:name: transformer-block
:alt: Transformer Block
:width: 50%
:align: center

Input flows through multi-head attention, layer normalization, feed-forward networks, and another normalization step.
```

Let us ignore the residual connection for now. The input is first passed through multi-head attention, followed by layer normalization. Then, the output of the normalization is passed through feed-forward networks and another layer normalization step.

### Residual Connection

```{figure} https://i.sstatic.net/UcJSa.png
:name: residual-connection
:alt: Residual Connection
:width: 30%
:align: center

Residual connection.
```

Now, let us consider the *residual connection*.
A residual connection, also known as a *skip connection*, is a technique used to stabilize the training of deep neural networks. More specifically, let us denote by $f$ the neural network that we want to train, which is the multi-head attention or feed-forward networks in the transformer block. The residual connection is defined as:

$$
\underbrace{x_{\text{out}}}_{\text{output}} = \underbrace{x_{\text{in}}}_{\text{input}} + \underbrace{f(x_{\text{in}})}_{\text{component}}.
$$

Note that rather than learning the complete mapping from input to output, the network $f$ learns to model the residual (difference) between them. This is particularly advantageous when the desired transformation approximates an identity mapping, as the network can simply learn to output values near zero.

Residual connections help prevent the vanishing gradient problem.
Deep learning models like LLMs consist of many layers, which are trained to minimize the loss function ${\cal L}_{\text{loss}}$ with respect to the parameters $\theta$.
To this end, the gradient of the loss function is computed using the chain rule as

$$
\frac{\partial {\cal L}_{\text{loss}}}{\partial \theta} = \frac{\partial {\cal L}_{\text{loss}}}{\partial f_L} \cdot \frac{\partial f_L}{\partial f_{L-1}} \cdot \frac{\partial f_{L-1}}{\partial f_{L-2}} \cdot ... \cdot \frac{\partial f_{l+1}}{\partial f_l} \cdot \frac{\partial f_l}{\partial \theta}
$$

where $f_i$ is the output of the $i$-th layer. The gradient vanishing problem occurs when the individual terms $\frac{\partial f_{i+1}}{\partial f_i}$ are less than 1. As a result, the gradient becomes smaller and smaller as the gradient flows backward through earlier layers.
By adding the residual connection, the gradient for the individual term becomes:

$$
\frac{\partial x_{i+1}}{\partial x_i} = 1 + \frac{\partial f_i(x_i)}{\partial x_i}
$$

Notice the "+1" term, which is the direct path from the input to the output. The chain rule is thus modified as:

$$\left(1 + \frac{\partial f_{L-1}}{\partial x_{L-1}}\right)\left(1 + \frac{\partial f_{L-2}}{\partial x_{L-2}}\right)\left(1 + \frac{\partial f_{L-3}}{\partial x_{L-3}}\right)...$$

When we expand this, we can group terms by their order (how many $\partial f_i$ terms are multiplied together):
We can write this more concisely using $O_n$ to represent terms of nth order:

$$1 + O_1 + O_2 + O_3 + ...$$

where:
- $O_1 = \frac{\partial f_{L-1}}{\partial x_{L-1}} + \frac{\partial f_{L-2}}{\partial x_{L-2}} + \frac{\partial f_{L-3}}{\partial x_{L-3}} + ...$
- $O_2 = \frac{\partial f_{L-1}}{\partial x_{L-1}}\frac{\partial f_{L-2}}{\partial x_{L-2}} + \frac{\partial f_{L-2}}{\partial x_{L-2}}\frac{\partial f_{L-3}}{\partial x_{L-3}} + \frac{\partial f_{L-1}}{\partial x_{L-1}}\frac{\partial f_{L-3}}{\partial x_{L-3}} + ...$
- $O_3 = \frac{\partial f_{L-1}}{\partial x_{L-1}}\frac{\partial f_{L-2}}{\partial x_{L-2}}\frac{\partial f_{L-3}}{\partial x_{L-3}} + ...$

Without the residual connection, we only have the $O_L$ terms for the network with $L$ layers, which is subject to the gradient vanishing problem. Whereas with the residual connection, we have the lower-order terms like $O_1, O_2, O_3, ...$ for the network with $L$ layers, which is less susceptible to the gradient vanishing problem.

```{admonition} Residual Connection
:class: tip

Residual connections are a architectural innovation that allows neural networks to be much deeper without degrading performance. It was proposed by He et al. {footcite:p}`he2015deep` for image processing from Microsoft Research.
```


```{admonition} Residual connection mitigates gradient explosion
:class: tip

Residual connections also help prevent gradient explosion, even though this may not be obvious from the chain rule perspective. As shown in {footcite:p}`philipp2017exploding`, the residual connection provides an alternative path for gradients to flow through. By distributing gradients between the residual path and the learning component path, the gradient is less likely to explode.
```

## Other miscellaneous components

### Position embedding

Position embedding is also an interesting component that is used to encode the position of the tokens in the sequence.
A key limitation of the attention mechanism is that it is *permutation invariant*.
This means that the order of the input tokens does not matter, e.g., "The cat sat on the mat" and "The mat sat on the cat" are the same.
To better capture the position information, transformers add to the input token embedding *a position embedding*.

To understand how this works, let us approach from a naive approach.
Suppose that we have a sequence of $T$ token embeddings, denoted by $x_1, x_2, ..., x_T$, each of which is a $d$-dimensional vector.
A simple way to encode the position information is to add a position index to each token embedding, i.e.,

$$
x_t := x_t + \beta t,
$$

where $t = 1, 2, ..., T$ is the position index of the token in the sequence, and $\beta$ is the step size. This appears to be simple but has a critical problem.

1. **Unbounded**: The position index can be arbitrarily large. When the models see a sequence longer than those in training data, it may suffer since the model will be exposed to a new position index that the model has never seen before.
2. **Discrete**: The position index is discrete, which means that the model cannot capture the position information in a smooth manner.

Because this naive approach has the problems, let us consider another approach. Let us represent the position index using a binary vector of length $d$. For example, in case of $d=4$, we have the following binary vectors:

$$
\begin{align*}
  0: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{0}} & &
  8: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{0}} \\
  1: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{1}} & &
  9: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{1}} \\
  2: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{0}} & &
  10: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{0}} \\
  3: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{1}} & &
  11: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{0}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{1}} \\
  4: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{0}} & &
  12: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{0}} \\
  5: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{1}} & &
  13: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{0}} \ \ \color{red}{\texttt{1}} \\
  6: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{0}} & &
  14: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{0}} \\
  7: \ \ \ \ \color{orange}{\texttt{0}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{1}} & &
  15: \ \ \ \ \color{orange}{\texttt{1}} \ \ \color{green}{\texttt{1}} \ \ \color{blue}{\texttt{1}} \ \ \color{red}{\texttt{1}} \\
\end{align*}
$$

Then, one may use the binary vector as the position embedding as follows:

$$
x_{t,i} := x_{t,i} + \text{Pos}(t, i),
$$

where $\text{Pos}(t, i)$ is the position embedding vector of the position index $t$ and the dimension index $i$.
This representation is good in the sense that it is unbounded. Yet, it is still discrete.

An elegant position embedding, which is used in transformers, is the *sinusoidal position embedding* {footcite:p}`vaswani2017attention`. It appears to be complicated but stay with me for a moment.

$$
\text{Pos}(t, i) =
\begin{cases}
\sin\left(\dfrac{t}{10000^{2i/d}}\right), & \text{if } i \text{ is even} \\
\cos\left(\dfrac{t}{10000^{2i/d}}\right), & \text{if } i \text{ is odd}
\end{cases},
$$

where $i$ is the dimension index of the position embedding vector. This position embedding is added to the input token embedding as:

$$
x_{t,i} := x_{t,i} + \text{Pos}(t, i),
$$

It appears to be complicated but it can be seen as a continuous version of the binary position embedding above. To see this, let us plot the position embedding for the first 100 positions.

```{figure} https://kazemnejad.com/img/transformer_architecture_positional_encoding/positional_encoding.png
:name: transformer-position-embedding
:alt: Transformer Position Embedding
:width: 80%
:align: center

The position embedding. The image is taken from https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
```

We note that, just like the binary position embedding, the sinusoidal position embedding also exhibits the alternating pattern (vertically) with frequency increasing as the dimension index increases (horizontal axis). Additionally, the sinusoidal position embedding is continuous, which means that the model can capture the position information in a smooth manner.

Another key property of the sinusoidal position embedding is that the dot similarity between the two position embedding vectors represent the distance between the two positions, regardless of the position index.

```{figure} https://kazemnejad.com/img/transformer_architecture_positional_encoding/time-steps_dot_product.png

:name: transformer-position-embedding-similarity
:alt: Transformer Position Embedding Similarity
:width: 80%
:align: center

The dot similarity between the two position embedding vectors represent the distance between the two positions, regardless of the position index. The image is taken from https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
```


```{admonition} Why additive position embedding?
:class: tip

The sinusoidal position embedding is additive, which alter the token embedding. Alternatively, one may concatenate, instead of adding, the position embedding to the token embedding, i.e., $x_{t,i} := [x_{t,i}; \text{Pos}(t, i)]$. This makes it easier for a model to distinguish the position information from the token information. So why not use the concatenation?

One reason is that the concatenation requires a larger embedding dimension, which increases the number of parameters in the model.
Instead, adding the position embedding creates an interesting effect in the attention mechanism.
Interested readers can check out [this Reddit post](https://www.reddit.com/r/MachineLearning/comments/cttefo/comment/exs7d08/?utm_source=reddit&utm_medium=web2x&context=3).

```

```{footbibliography}
:style: unsrt
:filter: docname in docnames
```
