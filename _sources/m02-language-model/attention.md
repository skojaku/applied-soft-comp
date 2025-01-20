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

# Attention: Learning to Focus

What if, like humans, our neural networks could learn to focus on what's important? When you read a sentence or look at a scene, you don't process everything with equal importance. You pay attention to specific parts at different times. This fundamental insight led to one of the most important innovations in deep learning: *attention mechanisms*.

## Why Attention is Needed

Consider this translation task:

"The cat sat on the mat because it was comfortable."

What does "it" refer to - the cat or the mat? As humans, we naturally link "it" to "cat" because we understand cats seek comfort. But traditional sequence models like vanilla RNNs and LSTMs struggle with such connections, especially in longer sequences.

Attention mechanisms allow models to focus on relevant parts of the input sequence while generating the output.
Instead of packing the information into a fixed-size memory (e.g., hidden state), the attention mechanism creates a matrix of attention weights within the given sequences. This weight is learned by a neural network that takes the corresponding variables as input.

[Figure: Visualization showing how attention "looks back" at input sequence while generating output]

## Mathematical Framework

Let's formalize this intuition. Given:
- An input sequence of $n$ vectors: $(x_1, ..., x_n)$
- Current decoder hidden state: $h_t$
- Encoder hidden states: $(h^{enc}_1, ..., h^{enc}_n)$

The attention mechanism computes:

1. Alignment scores $e_{tj}$ between the decoder state and each encoder state:
   $$e_{tj} = score(h_t, h^{enc}_j)$$

2. Attention weights through softmax normalization:
   $$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^n \exp(e_{tk})}$$

3. Context vector as weighted sum:
   $$c_t = \sum_{j=1}^n \alpha_{tj}h^{enc}_j$$

```{note}
The score function can take various forms:
- Dot product: $score(h_t, h^{enc}_j) = h_t^\top h^{enc}_j$
- Additive: $score(h_t, h^{enc}_j) = v^\top \tanh(W[h_t; h^{enc}_j])$
- Multiplicative: $score(h_t, h^{enc}_j) = h_t^\top W h^{enc}_j$
```

## Implementation Example

Let's implement a basic attention mechanism in PyTorch:

```{code-cell} ipython3
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # For additive attention
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]

        batch_size, seq_len, hidden_size = encoder_outputs.size()

        # Repeat hidden state seq_len times
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)

        # Calculate attention weights
        return F.softmax(attention, dim=1)
```

```{tip}
When implementing attention:
- Always check tensor dimensions carefully
- Use broadcasting to avoid explicit loops
- Consider numerical stability in softmax computation
- Monitor attention weights to ensure they sum to 1
```

## Visualizing Attention

One of the most powerful aspects of attention is its interpretability. The attention weights $\alpha_{tj}$ directly show us what parts of the input the model is focusing on at each step.

[Figure: Heatmap showing attention weights during translation, with x-axis as input words and y-axis as output words]

## Types of Attention

We've covered basic attention, but several variants exist:

1. Global vs Local Attention
   - Global: Attends to all source positions
   - Local: Only attends to a window of positions

2. Self-Attention
   - Allows sequence to attend to itself
   - Key component in modern architectures

```{note}
While we often visualize attention as "looking back" at the input, mathematically it's creating a weighted combination of values. This simple yet powerful idea has revolutionized sequence modeling.
```

## Exercises for Understanding

1. Why does attention help with the vanishing gradient problem?
2. Implement the dot-product version of the attention score function
3. Analyze how attention weights change with sequence length
4. Compare computation complexity of different attention variants

## Further Exploration

Consider these questions:
- How would you modify the attention mechanism for document summarization?
- What happens if we stack multiple attention layers?
- How might attention help in image captioning?

```{tip}
When experimenting with attention:
- Start with simple sequences to verify implementation
- Visualize attention weights frequently
- Try different score functions
- Monitor memory usage with long sequences
```

This lecture note has provided a foundation for understanding attention mechanisms. In practice, you'll find them indispensable for many sequence processing tasks, from translation to summarization to image captioning.