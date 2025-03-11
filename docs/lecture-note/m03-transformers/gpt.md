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

# Generative Pre-trained Transformer (GPT)

The Generative Pre-trained Transformer (GPT) {footcite}`radford2018language` represents a significant evolution in transformer-based language models, focusing on powerful text generation capabilities through a decoder-only architecture. While BERT uses bidirectional attention to understand context, GPT employs unidirectional (causal) attention to generate coherent text by predicting one token at a time.




```{admonition} GPT in interactive mode:
:class: tip

[Here is a demo notebook for GPT](https://static.marimo.app/static/gpt-ar61)

To run the notebook, download the notebook as a `.py` file and run it with:

```bash
marimo edit --sandbox gpt-interactive.py
```

You will need to install `marimo` and `uv` to run the notebook. But other packages will be installed automatically in uv's virtual environment.
```

## Architecture

Like in BERT, GPT also uses a transformer architecture. The main difference is that BERT uses an encoder transformer, while GPT uses a decoder transformer with some modifications.

```{figure} https://heidloff.net/assets/img/2023/02/transformers.png
:name: gpt-architecture
:alt: GPT architecture
:align: center
:width: 80%

GPT architecture.
```


```{tip}

The GPT model family has evolved through several iterations, starting with GPT-1 in 2018 which introduced the basic architecture with 117M parameters and transfer learning capabilities. GPT-2 followed in 2019 with 1.5B parameters and zero-shot abilities, while GPT-3 in 2020 dramatically scaled up to 175B parameters, enabling few-shot learning. The latest GPT-4 (2023) features multimodal capabilities, improved reasoning, and a 32K token context window. Throughout these iterations, the core decoder-only transformer architecture remained unchanged, with improvements coming primarily from increased scale that enabled emergent capabilities.

```{figure} https://miro.medium.com/v2/resize:fit:1400/1*Wnn0e8B-_IiTvmpv-1P7Iw.png
:name: gpt-evolution
:alt: GPT evolution
:align: center
:width: 80%

```

### Core Components

```{figure} https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7536a59a-5326-4a8b-ab12-cebe49acde31_1438x936.png
:name: gpt-causal-attention
:alt: GPT causal attention
:align: center
:width: 80%

Causal attention in GPT.
```

Like BERT, GPT uses learned token embeddings to convert input tokens into continuous vector representations. The model also employs learned positional embeddings that are added to the token embeddings to encode position information. A key difference from BERT is that GPT uses a *causal attention mechanism*, which means each position can only attend to previous positions in the sequence, enabling the model to generate text in a left-to-right fashion by predicting one token at a time.



### Causal Language Modeling

Causal (autoregressive) language modeling is the pre-training objective of GPT, where the model learns to predict the next token given all previous tokens in the sequence. More formally, given a sequence of tokens $(x_1, x_2, ..., x_n)$, the model is trained to maximize the likelihood:

$$
P(x_1, ..., x_n) = \prod_{i=1}^n P(x_i|x_1, ..., x_{i-1})
$$

For example, given the partial sentence "The cat sat on", the model learns to predict the next word by calculating probability distributions over its entire vocabulary. During training, it might learn that "mat" has a high probability in this context, while "laptop" has a lower probability.

```{note}
While BERT uses bidirectional attention and sees the entire sequence at once (making it powerful for understanding), GPT's unidirectional approach more naturally models how humans write text, i.e., one word at a time, with each word influenced by all previous words.
The bidirectional nature of BERT is more powerful for understanding, but it is less suitable for text generation.
```

```{tip}
The autoregressive nature of GPT means it's particularly sensitive to the initial tokens (prompt) it receives. Well-crafted prompts that establish clear patterns or constraints can significantly improve generation quality.
```

The next-token prediction objective has remained unchanged across all GPT versions due to its remarkable effectiveness. Rather than modifying this core approach, improvements have come from increasing model size and refining the architecture. This simple yet powerful training method has become fundamental to modern language models.

```{admonition} Scaling Laws
:class: tip
:name: scaling-laws

Language model performance improves *predictably* as models get larger, following simple mathematical relationships (power laws). The larger the model, the better it performs - and this improvement is reliable and measurable. This predictability was crucial for the development of models like GPT-3 and Claude, as it gave researchers confidence that investing in larger models would yield better results.
Importantly, larger models are more efficient learners - they need proportionally less training data and fewer training steps to achieve good performance.

These findings revolutionized AI development by showing that better AI systems could be reliably built simply by scaling up model size, compute, and data in the right proportions. This insight led directly to the development of increasingly powerful models, as researchers could confidently invest in building larger and larger systems knowing they would see improved performance.

See the paper [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) for more details.

```{figure} https://miro.medium.com/v2/resize:fit:1400/1*5fsJPwvFjS7fo8g8NwsxNA.png
:name: scaling-laws-figure
:alt: Scaling laws figure
:align: center
:width: 80%
```

## Inference Strategies

GPT does not generate text in one go. Instead, it predicts the next token repeatedly to generate text. GPT does not pick a specific token but provides a *probability distribution* over the next token. It is our job to *sample* a token from the distribution. There are several strategies to sample a token from the distribution as we will see below.

```{figure} https://media.licdn.com/dms/image/v2/D4E22AQFZFRSwwzCSqQ/feedshare-shrink_2048_1536/feedshare-shrink_2048_1536/0/1725003016027?e=2147483647&v=beta&t=oBH1s4V8N0wKCOJQakA_wrwgFrixs56S0s_QafZOvbA
:name: gpt-inference
:alt: GPT inference
:align: center
:width: 50%

GPT predicts the next token repeatedly to generate text.
```

### Greedy and Beam Search

When generating text, language models assign probabilities to possible next tokens.
Sampling a token from the distribution is not as easy as it might seem. This is because the distribution is high-dimensional. Namely, we need to sample a single token from millions of possible tokens, and thus, sampling a token can be computationally very expensive.

**Greedy sampling** always picks the highest probability token, which is deterministic but can lead to repetitive or trapped text. For example, if the model predicts "the" with high probability, it will always predict "the" again.

```{figure} https://huggingface.co/blog/assets/02_how-to-generate/greedy_search.png
:name: gpt-greedy-search
:alt: GPT greedy search
:align: center
:width: 50%

GPT greedy search.
```

**Beam search** alleviates this problem by taking into account the high-order dependencies between tokens. For example, in generating "The cat ran across the ___", beam search might preserve a path containing "mat" even if "floor" or "room" have higher individual probabilities at that position. This is because the complete sequence like "mat quickly" could be more probable when considering the token next after "mat".  "The cat ran across the mat quickly" is a more natural phrase than "The cat ran across the floor quickly" when considering the full flow and common linguistic patterns.


 ```{figure} https://huggingface.co/blog/assets/02_how-to-generate/beam_search.png
 :name: gpt-beam-search
 :alt: GPT beam search
 :align: center
 :width: 50%

 GPT beam search.
 ```

Beam search maintains multiple possible sequences (beams) in parallel, exploring different paths simultaneously. At each step, it expands all current beams, scores the resulting sequences, and keeps only the top-k highest scoring ones. For instance, with a beam width of 3:
- First beams might be: ["The cat ran", "The cat walked", "The cat jumped"]
- Next step: ["The cat ran across", "The cat ran through", "The cat walked across"]
- And so on, keeping the 3 most promising complete sequences at each step

This process continues until reaching the end, finally selecting the sequence with highest overall probability. The beam search can be combined with top-k sampling or nucleus sampling. For example, one can sample a token based on the top-k sampling or nucleus sampling to form the next beam.

While beam search often produces high-quality outputs since it considers longer-term coherence, it can still suffer from the problem of repetitive or trapped text.

### From Deterministic to Stochastic Sampling

Both greedy and beam search are deterministic. They pick the most likely token at each step. However, this creates a loop where the model always predicts the same tokens repeatedly. A simple way to alleviate this problem is to sample a token from the distribution.

**Top-k Sampling** relaxes the deterministic nature of greedy sampling by selecting randomly from the k most likely next tokens at each generation step. While this introduces some diversity compared to greedy sampling, choosing a fixed k can be problematic. Value of $k$ might be too large for some distribution tails (including many poor options) or too small for others (excluding reasonable options).

**Nucleus Sampling**~{footcite}`holtzman2019curious` addresses this limitation by dynamically selecting tokens based on cumulative probability. It samples from the smallest set of tokens whose cumulative probability exceeds a threshold p (e.g. 0.9). This adapts naturally to different probability distributions, i.e., selecting few tokens when the distribution is concentrated and more when it's spread out. This approach often provides a good balance between quality and diversity.

```{figure} https://storage.googleapis.com/zenn-user-upload/8p2r9urhtn5nztdg6mnia3toibhl
:name: gpt-top-k-top-p
:alt: GPT top-k top-p
:align: center
:width: 80%

Nucleus sampling. The image is taken from [this blog](https://zenn.dev/hellorusk/articles/1c0bef15057b1d).
```

**Temperature Control**
Temperature ($\tau$) modifies how "concentrated" the probability distribution is for sampling by scaling the logits before applying softmax:

$$
p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}
$$

where $z_i$ are the logits and $\tau$ is the temperature parameter. Lower temperatures ($\tau < 1.0$) make the distribution more peaked, making high probability tokens even more likely to be chosen, leading to more focused and conservative outputs. Higher temperatures ($\tau > 1.0$) flatten the distribution by making the logits more similar, increasing the chances of selecting lower probability tokens and producing more diverse but potentially less coherent text. As $\tau \to 0$, the distribution approaches a one-hot vector (equivalent to greedy search), while as $\tau \to \infty$, it approaches a uniform distribution.


```{figure} https://cdn.prod.website-files.com/618399cd49d125734c8dec95/6639e35ce91c16b3b9564b2f_mxaIPcROZcBFYta1I0nzWjlGTgs-LxzUOE3p6Kbvf9qPpZzBh5AAZG7ciRtgVquhLTtrM8ToJdNd-ubXvuz8tRfrqBwSozWHCj457pm378buxz2-XrMfWzfSv3b793QP61kLxRKT299WP1gbas_E118.png
:name: gpt-temperature
:alt: GPT temperature
:align: center
:width: 80%

Temperature controls the concentration of the probability distribution. Lower temperature makes the distribution more peaked, while higher temperature makes the distribution more flat.
```


## Hands-on

Let us learn how to generate text with GPT-2. We will use the `transformers` library to load the model and tokenizer.

```{code-cell} ipython
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# Force CPU usage and clear memory
torch.cuda.empty_cache() if torch.cuda.is_available() else None
gc.collect()

# Force CPU usage for this example
if torch.backends.mps.is_available():
    torch_device = "mps"  # MPS (Apple Silicon)
elif torch.cuda.is_available():
    torch_device = "cuda" #  CUDA (GPU)
else:
    torch_device = "cpu" # (CPU)

# Initialize tokenizer and set pad token
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Initialize model and set pad token id
model = AutoModelForCausalLM.from_pretrained("gpt2").to(torch_device)
model.config.pad_token_id = model.config.eos_token_id

# We encode the input text into tokens.
model_inputs = tokenizer('I enjoy walking with ', return_tensors='pt').to(torch_device)

# generate 40 new tokens
greedy_output = model.generate(**model_inputs,
                             max_new_tokens=20,
                             pad_token_id=tokenizer.eos_token_id)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
```

The greedy search approach often falls into repetitive patterns. This limitation occurs because greedy search always selects the most probable next token, leading to predictable and sometimes monotonous outputs.
Interested readers can refer to these papers {footcite}`vijayakumar2016diverse` and {footcite}`shao2017generating` for more details.

Now, let us showcase how to generate text with beam search.

```{code-cell} ipython
beam_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    num_beams=5, # Number of beams
    early_stopping=True # Stop when the model generates the EOS token
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
```

Beam search produces more polished text compared to greedy search by exploring multiple possible sequences simultaneously. However, it still tends to generate repetitive content since it focuses on the most probable paths.

Beam search generates multiple sequences in parallel, and we can retrieve the top-k sequences with the highest probability as follows:

```{code-cell} ipython
# set return_num_sequences > 1
beam_outputs = model.generate(
    **model_inputs,
    max_new_tokens=40,
    num_beams=5, # Number of beams
    num_return_sequences=5, # Number of output sequences
    early_stopping=True # Stop when the model generates the EOS token
)

# now we have 3 output sequences
print("Output:\n" + 100 * '-')
for i, beam_output in enumerate(beam_outputs):
  print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))
```

Notice how similar these generated sequences are to each other. This similarity stems from beam search's fundamental approach: it still favors highly probable sequences, which often share common patterns and phrasings. This behavior reveals a key limitation of probability-based selection strategies.

These results highlight a crucial insight: the most probable sequences aren't necessarily the most engaging or human-like. Research has shown that human-written text actually exhibits more variety and unpredictability than the highest-probability sequences generated by language models {footcite}`holtzman2019curious`.

This observation can be interpreted in two ways: either our models aren't yet sophisticated enough to capture the true distribution of human text {footcite}`welleck2020unlikelihood`, or our generation strategies need to incorporate some randomness to better mimic human writing patterns {footcite}`holtzman2019curious`. Let's explore the second hypothesis by introducing controlled randomness into our generation process.

In transformers, we set `do_sample=True` to activate sampling. And we set `top_k=50` to perform the top-k sampling, with $k=50$.

```{code-cell} ipython
from transformers import set_seed
# set seed to reproduce results. Feel free to change the seed though to get different results
set_seed(42)

# activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=50
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

Now the generated text becomes very different from the greedy search. We can control the "randomness" of the generated text by setting the temperature. For example, by default, the temperature is 1.0. But we can reduce it to make the generated text to have a higher probability.


```{code-cell} ipython
set_seed(42)

# activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=50,
    temperature=0.7
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

Now, let us try nucleus sampling. We set `top_p=0.95` to perform the nucleus sampling, with $p=0.95$.


```{code-cell} ipython
set_seed(42)

# activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_p=0.95,
    top_k=0, # deactivate top_k.
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

It is hard to say which method is better. In practice, both methods often work well. You can combine both methods by setting `top_k=50` and `top_p=0.95`, for instance. This leaves the candidate set of tokens that are more likely than those in the top-k sampling or nucleus sampling. So let's sample multiple sequences using the nucleus sampling and top-k sampling being toggled.

```{code-cell} ipython
set_seed(42)

# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
sample_outputs = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3,
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
```


## 🔥 Exercise 🔥

Write an interesting story using only GPT-2. Write the first prompt and let GPT-2 generate the rest.

Change the temperature, top-k, top-p, and other parameters to make the story natural and interesting.

