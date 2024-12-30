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

# Breaking the Code: From Enigma to Sequence-to-Sequence Models

```{figure} https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/%27bombe%27.jpg/1500px-%27bombe%27.jpg
:alt: Bombe
:width: 100%

Bombe, the codebreaking machine used by Alan Turing and his team at Bletchley Park.

```

During World War II, the German military used a sophisticated encryption device called the Enigma machine. This mechanical marvel could transform readable messages into seemingly random sequences of letters, which could only be deciphered by another Enigma machine set to the same configuration. Breaking this code was considered nearly impossible, since the machine could be configured in millions of different ways, and the settings were changed daily.

Alan Turing and his team at Bletchley Park realized that breaking Enigma required understanding how one sequence (the encrypted message) mapped to another sequence (the original text). Their groundbreaking work not only helped win the war but also laid the foundation for modern computing and, in many ways, foreshadowed one of the most powerful concepts in modern machine learning: *sequence-to-sequence transformation*.

```{note}
The challenge faced at Bletchley Park was, in essence, a sequence-to-sequence problem: transforming a sequence of encrypted characters back into their original message. While the methods used were very different from today's neural networks, the fundamental goal was the same.
```

Today's sequence-to-sequence (seq2seq) models {footcite:p}`sutskever2014sequence` tackle similar challenges, though at a far more sophisticated level. Like the codebreakers at Bletchley Park, these models learn to transform one sequence into another, whether it's translating languages, converting speech to text, or summarizing documents. The key difference is that instead of relying on mechanical rotors and manual computations (which are cool!!), modern seq2seq models use neural networks to learn these transformations automatically from data.

## Model architecture

```{figure} https://raw.githubusercontent.com/bentrevett/pytorch-seq2seq/eff9642693c3d83c497a791e80a34e740874f5cd/assets/seq2seq7.png
:alt: seq2seq model architecture
:width: 100%

seq2seq model architecture.
```

seq2seq models are based on a two-part architecture: the encoder and decoder.

- **Encoder** reads the input sequence and compresses it into a context vector, which captures the meaning and nuances of the input. The encoder processes the input sequence one by one, updating its internal state at each step. The final state becomes a context vector that contains a compressed version of the entire input sequence.
- **Decoder** takes this fixed-size context vector and generates a completely new sequence autoregressively, potentially of different length and in a different format altogether.


## Mathematical Framework

Sequence-to-sequence models work by transforming sequences using probabilities. At their core, they ask: "Given this input sequence, what's the probability of generating each element in the output sequence?" During training, the model learns these probabilities by seeing many examples of input-output pairs. Then during inference, it uses what it learned to generate the most likely output sequence for a new input.

Specifically, seq2seq model learns to model the conditional probability between input sequence $(x_1,...,x_n)$ of length $n$ and output sequence $(y_1,...,y_m)$ of length $m$. This probability is expressed as:

$$
P(y_1,...,y_m|x_1,...,x_n)
$$

The encoder processes the input sequence to create hidden states through the function $h_t = f(x_t, h_{t-1})$, where $f$ is typically an RNN function like LSTM or GRU. The final hidden state $h_n$ becomes the context vector $c$.

The decoder generates the output sequence by modeling each element as $P(y_t|y_1,...,y_{t-1},c)$, effectively decomposing the joint probability using the chain rule:

$$
P(y_1,...,y_m|x_1,...,x_n) = \prod_{t=1}^m P(y_t|y_1,...,y_{t-1},c)
$$

The decoder updates its hidden state at each timestep $t$ using $s_t = g(y_{t-1}, s_{t-1}, c)$, where $g$ is another RNN function.

```{note}
The decoder takes two hidden states as input: the previous hidden state $s_{t-1}$ and the context vector $c$. Context vector $c$ is crucial for decoder to note the information from the encoder.
```

## Limitation of seq2seq

The traditional seq2seq architecture faces several critical limitations.The most significant challenge is the **information bottleneck**: compressing the entire input sequence into a fixed-size context vector $c$. This becomes particularly problematic for *long sequences*, where crucial information may be lost during compression.

A second major limitation is the **long-range dependencies** and the *vanishing gradient problem*. The model struggles to maintain relationships between distant elements in long sequences, as gradients become increasingly small during backpropagation through time, even with LSTM. This particularly hinders the learning of the earlier parts of input sequences, resulting in degraded performance for longer inputs.

A third limitation is that the model treats all input elements equally when creating the context vector, despite the fact that *not all inputs are equally relevant* for each output element.


## Attention Mechanism

```{figure} https://lena-voita.github.io/resources/lectures/seq2seq/attention/attn_for_steps/6-min.png
:alt: attention mechanism
:width: 100%

The attention mechanism. The decoder can now see the output of the encoder at each step. Attention mechanism learns the "attention" the decoder should pay to the encoder at each step.
```

The attention mechanism solves these limitations by letting the decoder focus on specific parts of the input sequence as needed. Instead of using just one fixed context vector, the decoder can look back at different input elements while generating each output. It does this by calculating attention weights that show how important each input element is at each step. These weights are learned during training, so that the model can automatically figure out which parts of the input matter most when generating the output.


```{note}
The attention mechanism was first introduced in neural machine translation but has since become a fundamental component in many deep learning architectures, including the transformer model that powers systems like GPT and BERT.
```


The attention mechanism works as follows. It first calculates the "unnormalized" attention weights $e_{tj}$ for each input $j$ at each step $t$ using a scoring function $a$, which can be a neural network or a simple dot product between the decoder hidden state $s_{t-1}$ and the encoder hidden state $h_j$.

This "unnormalized" attention weights are then normalized using the softmax function to obtain the "normalized" attention weights $\alpha_{tj}$:

$$
\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^n \exp(e_{tk})}
$$

The context vector $c_t$ is then computed as a weighted sum of the encoder hidden states $h_j$ using the attention weights $\alpha_{tj}$:

$$
c_t = \sum_{j=1}^n \alpha_{tj}h_j
$$

The training process minimizes the negative log-likelihood loss:

$$
\mathcal{L} = -\sum_{t=1}^m \log P(y_t|y_1,...,y_{t-1},x_1,...,x_n)
$$


```{note}
Training a seq2seq model is a fun but challenging coding exercise that covers many technical topics in deep learning, such as padding, masking, teacher forcing, and more. While we will not cover these topics in this course due to technical complexity, I highly recommend you to try it out on your own.
You can find good tutorials on seq2seq model implementation in [this blog](https://jaketae.github.io/study/seq2seq-attention/), [this blog](https://greydanus.github.io/2017/01/07/enigma-rnn/), and [this repo](https://github.com/hkhoont/scale_ai_engima_machine).
```


```{footbibliography}
:style: unsrt
```