import torch
import torch.nn as nn
from typing import Tuple


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.tanh = nn.Tanh()

        self.to(device)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process each timestep in the sequence
        if input.dim() == 2:
            batch_size, input_size = input.size()
            seq_length = 1
            input = input.unsqueeze(1)  # Add sequence dimension
        else:
            batch_size, seq_length, input_size = input.size()

        outputs = torch.zeros(batch_size, seq_length, self.i2o.out_features, device=self.device)

        # Ensure hidden has shape (batch_size, hidden_size)
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)

        # Match batch sizes
        if batch_size != hidden.size(0):
            hidden = hidden.expand(batch_size, -1)

        # Process sequence
        for t in range(seq_length):
            # Get current input timestep
            current_input = input[:, t, :]

            # Combine input and hidden state
            combined = torch.cat((current_input, hidden), 1)

            # Update hidden state and get output
            hidden = self.tanh(self.i2h(combined))
            output = self.i2o(combined)

            # Store output for this timestep
            outputs[:, t, :] = output

        if seq_length == 1:
            outputs = outputs.squeeze(1)

        return outputs, hidden

    def initHidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=self.device)


class SimpleLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(SimpleLSTM, self).__init__()
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

        #if seq_length == 1:
        #    outputs = outputs.squeeze(1)

        return outputs, (h_t, c_t)

    def initHidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.hidden_size, device=self.device),
            torch.zeros(batch_size, self.hidden_size, device=self.device),
        )