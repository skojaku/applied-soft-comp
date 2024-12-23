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

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure input has shape (batch_size, input_size)
        if input.dim() == 1:
            input = input.unsqueeze(0)

        # Ensure hidden has shape (batch_size, hidden_size)
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)

        # Match batch sizes
        if input.size(0) != hidden.size(0):
            hidden = hidden.expand(input.size(0), -1)

        combined = torch.cat((input, hidden), 1)
        hidden = self.tanh(self.i2h(combined))
        output = self.i2o(combined)
        return output, hidden

    def initHidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=self.device)
