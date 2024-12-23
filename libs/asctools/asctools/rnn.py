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
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        hidden = self.tanh(hidden)
        return output, hidden

    def initHidden(self) -> torch.Tensor:
        return torch.zeros(1, self.hidden_size, device=self.device)
