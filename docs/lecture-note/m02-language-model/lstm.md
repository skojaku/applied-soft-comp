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


# Long Short-Term Memory (LSTM)

When rememberig a long story, you would naturally focus on important details while letting less relevant information fade. This selective memory is exactly what Long Short-Term Memory (LSTM) networks aim to achieve in artificial neural networks. While standard RNNs struggle with long-term dependencies due to the vanishing gradient problem, LSTMs offer a sophisticated solution through controlled memory mechanisms.

```{figure} ../figs/lstm.jpg
---
width: 500px
name: lstm
---

LSTM architecture showing the cell state (horizontal line at top) and the three gates: forget gate, input gate, and output gate. The cell state acts as a conveyor belt carrying information forward, while gates control information flow.

```


At the heart of an LSTM lies a *memory cell* (or *cell state*, i.e., $c_{t}$) that can maintain information over long periods. Think of this cell as a conveyor belt that runs straight through the network, allowing information to flow forward largely unchanged. This cell state forms the backbone of the LSTM's memory system.


```{figure} ../figs/lstm-forget-gate.jpg
---
width: 400px
name: lstm-01
align: center
---

Forget gate. $\sigma(x_t, h_t)$ decides how much of the previous cell state $c_{t-1}$ to keep. For example, if $\sigma(x_t, h_t) = 0$, the forget gate will completely forget the previous cell state. If $\sigma(x_t, h_t) = 1$, the forget gate will keep the previous cell state. $\sigma$ is the sigmoid function which is bounded between 0 and 1.
```

**Forget Gate:**
The LSTM controls this memory through three specialized neural networks called *gates*. The *forget gate* examines the current input and the previous hidden state to decide what information to remove from the cell state. Like a selective eraser, it outputs values between 0 and 1 for each number in the cell state, where 0 means "completely forget this" and 1 means "keep this entirely."


```{figure} ../figs/lstm-input-gate.jpg
---
width: 400px
name: lstm-02
align: center
---

Input gate. $\sigma(x_t, h_t)$ decides how much of the new information (that passes through the tanh function) to add to the cell state. For example, if $\sigma(x_t, h_t) = 0$, the input gate will completely ignore the new candidate information. If $\sigma(x_t, h_t) = 1$, the input gate will add the new candidate information to the cell state.
```

The input gate works together with a candidate memory generator to decide what new information to store. The input gate determines how much of the new candidate values should be added to the cell state, while the candidate memory proposes new values that could be added. This mechanism allows the network to selectively update its memory with new information.


```{figure} ../figs/lstm-output-gate.jpg
---
width: 400px
name: lstm-03
align: center
---
Output gate. $\sigma(x_t, h_t)$ decides how much of the cell state to reveal as output. For example, if $\sigma(x_t, h_t) = 0$, the output gate will completely hide the cell state. If $\sigma(x_t, h_t) = 1$, the output gate will reveal the cell state.
```

Finally, the output gate controls what parts of the cell state should be revealed as output. It applies a filtered version of the cell state to produce the hidden state, which serves as both the output for the current timestep and part of the input for the next timestep.

```{note}
The key innovation of LSTMs is not just having memory, but having controlled memory. The network learns what to remember and what to forget, rather than trying to remember everything.
```

## Mathematical Framework

The LSTM's operation can be described through a series of equations that work together to process sequential data. The cell state $C_t$ evolves according to:

$$ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t $$

where $f_t$ is the forget gate, $i_t$ is the input gate, and $\tilde{C}_t$ is the candidate memory. The $\odot$ symbol represents element-wise multiplication, allowing the gates to control information flow by scaling values between 0 and 1.

The gates themselves are neural networks that take the current input $x_t$ and previous hidden state $h_{t-1}$ as inputs:

$$ f_t = \sigma(W_f[h_{t-1}, x_t] + b_f) $$
$$ i_t = \sigma(W_i[h_{t-1}, x_t] + b_i) $$
$$ o_t = \sigma(W_o[h_{t-1}, x_t] + b_o) $$

The candidate memory is generated similarly:

$$ \tilde{C}_t = \tanh(W_c[h_{t-1}, x_t] + b_c) $$

Finally, the hidden state is produced by:

$$ h_t = o_t \odot \tanh(C_t) $$

While we've covered the basic equations of LSTMs, let's explore the technical details more thoroughly. When we write $[h_{t-1}, x_t]$ in our equations, we're performing vector concatenation, combining the previous hidden state with our current input. This creates a rich representation that helps the network make decisions about its memory. The weight matrices in our equations ($W_f$, $W_i$, $W_o$, and $W_c$) transform this concatenated input into the appropriate dimensions for each gate. For instance, if we have an input dimension of d and a hidden state dimension of h, these weight matrices will have dimensions $h × (h+d)$, ensuring our outputs maintain the correct size throughout the network.

## Common Challenges and Solutions

While LSTMs are powerful, they come with their own set of challenges. Despite being designed to handle the vanishing gradient problem better than vanilla RNNs, extremely long sequences can still pose difficulties. Practitioners often employ gradient clipping to maintain stable training. Memory consumption can become a bottleneck with very long sequences, but this can be addressed through techniques like truncated backpropagation or sequence chunking.

Overfitting is another common challenge, as LSTMs have numerous parameters to tune. To combat this, consider using dropout between LSTM layers, implementing layer normalization, or reducing model size if your task doesn't require the full complexity. Training speed can also be a concern due to the sequential nature of processing. Utilizing mini-batching and GPU acceleration can help, or you might consider using [Gated Recurrent Units (GRUs)](https://en.wikipedia.org/wiki/Gated_recurrent_unit) as a lighter alternative.

## Hands-on Implementation

Let us implement a simple LSTM model. Here is the code:

```{code-cell} ipython3
import torch
import torch.nn as nn
from typing import Tuple

class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        # Linear layers for gates
        combined_dim = input_size + hidden_size
        self.forget_gate = nn.Linear(combined_dim, hidden_size)
        self.input_gate = nn.Linear(combined_dim, hidden_size)
        self.cell_gate = nn.Linear(combined_dim, hidden_size)
        self.output_gate = nn.Linear(combined_dim, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)

        # Initialize forget gate bias to 1
        self.forget_gate.bias.data.fill_(1.0)

        self.to(device)

    def forward(
        self,
        input: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Process each timestep in the sequence
        if input.dim() == 2:
            batch_size, input_size = input.size()
            seq_length = 1
            input = input.unsqueeze(1)  # Add sequence dimension
        else:
            batch_size, seq_length, input_size = input.size()

        outputs = torch.zeros(batch_size, seq_length, self.i2o.out_features, device=self.device)

        # Unpack hidden state
        h_t, c_t = hidden

        # Ensure hidden states have shape (batch_size, hidden_size)
        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(0)
        if c_t.dim() == 1:
            c_t = c_t.unsqueeze(0)

        # Match batch sizes
        if batch_size != h_t.size(0):
            h_t = h_t.expand(batch_size, -1)
            c_t = c_t.expand(batch_size, -1)

        # Process sequence
        for t in range(seq_length):
            # Get current input timestep
            current_input = input[:, t, :]

            # Concatenate input and previous hidden state
            combined = torch.cat((current_input, h_t), dim=1)

            # Calculate gates
            f_t = torch.sigmoid(self.forget_gate(combined))
            i_t = torch.sigmoid(self.input_gate(combined))
            c_tilde = torch.tanh(self.cell_gate(combined))
            o_t = torch.sigmoid(self.output_gate(combined))

            # Update cell state and hidden state
            c_t = f_t * c_t + i_t * c_tilde
            h_t = o_t * torch.tanh(c_t)

            # Calculate output for this timestep
            outputs[:, t, :] = self.i2o(h_t)

        if seq_length == 1:
            outputs = outputs.squeeze(1)

        return outputs, (h_t, c_t)

    def initHidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.hidden_size, device=self.device),
            torch.zeros(batch_size, self.hidden_size, device=self.device),
        )
```

To showcase the LSTM, let's consider a toy task, i.e., parentheses matching. In this task, we are given a sequence of characters, where each character is either a parenthesis or a regular character. We want to predict whether the parentheses are matched or not. For example, the sequence `(a(b)c)` is valid, while the sequence `(a(b)c` is invalid.

```{code-cell} ipython3
from asctools.dataset import generate_parentheses_dataset

sequences, y_valid = generate_parentheses_dataset(n_samples=1000, min_length=25, max_length=25)

print("sequences[0]:", sequences[0])
print("y_valid[0]:", y_valid[0])
```

The LSTM model cannot directly take alphabet as input. Instead, we need to convert the alphabet to one-hot encoding.

```{code-cell} ipython3
import torch

def to_one_hot(sequence):
    alphabet = 'abcdefghijklmnopqrstuvwxyz)('
    one_hot = torch.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        one_hot[i, alphabet.index(sequence[i])] = 1
    return one_hot

sequences_one_hot = torch.stack([to_one_hot(sequence) for sequence in sequences], dim = 0)

```

We also need to convert the target to a tensor of the proper shape.

```{code-cell} ipython3
y_valid = torch.tensor(y_valid, dtype=torch.long)  # Shape should be (batch_size,)
```

Now, we are ready to train the LSTM model.

```{code-cell} ipython3
from asctools.rnn_trainer import RNNTrainer
from torch import nn
vocab_size = 28 # 26 characters + 2 parentheses

lstm = LSTM(input_size=vocab_size, hidden_size=32, output_size=2)
lstm.train()
trainer = RNNTrainer(lstm)
losses = trainer.train(
    input_tensors=sequences_one_hot, # This is the input sequence.
    targets=y_valid, # This is the target sequence.
    criterion=nn.CrossEntropyLoss(), # This is the loss function.
    max_epochs=300, # This is the maximum number of epochs.
    learning_rate=0.01, # This is the learning rate.
    clip_grad_norm=1.0, # This is to prevent the gradient from exploding or vanishing.
)
```


```{note}
`nn.CrossEntropyLoss()` is a loss function that is commonly used for classification tasks. It combines `nn.LogSoftmax()` and `nn.NLLLoss()`. See [here](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) for more details.
```

Let us confirm that the training loss is decreasing.

```{code-cell} ipython3
import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss during training')
plt.show()
```

Using the trained LSTM model, we can now evaluate its performance on the test set.

```{code-cell} ipython3
import numpy as np

lstm.eval()
eval_sequences, eval_y_valid = generate_parentheses_dataset(n_samples = 300)
eval_sequences_one_hot = torch.stack([to_one_hot(sequence) for sequence in eval_sequences], dim = 0)
outputs = []
for sequence in eval_sequences_one_hot:
    hidden = lstm.initHidden()
    for i in range(len(sequence)):
        output, hidden = lstm(sequence[i], hidden)

    # Prediction
    pred = torch.argmax(output, dim=1)
    outputs.append(pred.item())

accuracy = np.sum(np.array(outputs) == np.array(eval_y_valid)) / len(eval_y_valid)
print(f"Accuracy: {accuracy:.2f}")
```

## 🔥 Exercise 🔥

1. Make the problem more challenging by increasing the sequence length to 100.
2. Try using a simple RNN model by importing `RNN` from `asctools.rnn`, and compare the performance with the LSTM model.
3. The LSTM model uses a linear layer for producing the output (i.e., `self.i2o`). We can change it to a more complex, powerful function, such as a multilayer perceptron. Try implementing it by using `nn.Sequential`, e.g., `nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))`.