import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple, Callable
from torch.optim import Optimizer
from torch.nn import Module

class RNNTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Generic trainer for sequence models (RNN, LSTM, etc.)

        Args:
            model: The sequence model to train
            device: Device to use for training
        """
        self.model = model
        self.device = device
        self.model.to(device)

    def train(
        self,
        input_tensors: Union[torch.Tensor, List[torch.Tensor]],
        task: str = 'next_token',
        labels: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        criterion: Optional[Module] = None,
        optimizer: Optional[Optimizer] = None,
        max_epochs: int = 1000,
        learning_rate: float = 0.01,
        teacher_forcing_ratio: float = 0.5,
        patience: int = 10,
        min_delta: float = 1e-4,
        hidden_init_func: Optional[Callable] = None
    ) -> List[float]:
        """
        Train the sequence model

        Args:
            input_tensors: Input sequence tensor(s)
            task: 'next_token' or 'classification'
            labels: Class labels for classification task
            targets: Targets for next token prediction
            criterion: Loss function
            optimizer: Optimizer
            max_epochs: Maximum number of training epochs
            learning_rate: Learning rate for optimizer
            teacher_forcing_ratio: Probability of using teacher forcing
            patience: Number of epochs to wait for improvement before early stopping
            min_delta: Minimum change in loss to qualify as an improvement
            hidden_init_func: Function to initialize hidden state

        Returns:
            losses: List of losses during training
        """
        # Prepare inputs
        if isinstance(input_tensors, torch.Tensor):
            input_tensors = [input_tensors]
        input_tensors = [t.to(self.device) for t in input_tensors]

        if labels is not None:
            labels = labels.to(self.device)

        if targets is not None:
            targets = targets.to(self.device)

        criterion = criterion or nn.CrossEntropyLoss()
        optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        losses = []
        best_loss = float('inf')
        patience_counter = 0

        # Add task validation at the beginning
        if task not in ['next_token', 'classification']:
            raise ValueError(f"Invalid task '{task}'. Must be 'next_token' or 'classification'")

        if task == 'next_token' and targets is None:
            raise ValueError("Targets must be provided for next_token task")

        if task == 'classification' and labels is None:
            raise ValueError("Labels must be provided for classification task")

        for epoch in range(max_epochs):
            epoch_loss = 0

            for idx, input_tensor in enumerate(input_tensors):
                hidden = hidden_init_func() if hidden_init_func else None
                optimizer.zero_grad()

                if task == 'next_token':
                    sequence_loss = self._train_next_token(
                        input_tensor,
                        targets[:, idx] if targets.dim() > 1 else targets,
                        hidden,
                        criterion,
                        teacher_forcing_ratio
                    )
                else:  # Classification task
                    sequence_loss = self._train_classification(
                        input_tensor, hidden, criterion, labels[idx]
                    )

                sequence_loss.backward()
                optimizer.step()
                epoch_loss += sequence_loss.item()

            avg_loss = epoch_loss / len(input_tensors)
            losses.append(avg_loss)

            # Early stopping check
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{max_epochs}, Average Loss: {avg_loss:.4f}')

        return losses

    def _train_next_token(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        hidden: Optional[torch.Tensor],
        criterion: Module,
        teacher_forcing_ratio: float
    ) -> torch.Tensor:
        sequence_loss = 0
        output = None

        # Ensure input_tensor is 3D: [sequence_length, batch_size, feature_size]
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(1)

        for i in range(len(input_tensor) - 1):
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

            if use_teacher_forcing or i == 0:
                curr_input = input_tensor[i]
            else:
                # Create one-hot encoding from previous output
                curr_input = torch.zeros_like(input_tensor[0])
                curr_input.scatter_(1, output.argmax(1).unsqueeze(1), 1)

            if hidden is not None:
                output, hidden = self.model(curr_input, hidden)
            else:
                output = self.model(curr_input)

            # Ensure target is properly shaped for criterion
            target = target_tensor[i]
            if target.dim() == 0:  # If it's a scalar tensor
                target = target.unsqueeze(0)
            sequence_loss += criterion(output, target)

        return sequence_loss

    def _train_classification(
        self,
        input_tensor: torch.Tensor,
        hidden: Optional[torch.Tensor],
        criterion: Module,
        label: torch.Tensor
    ) -> torch.Tensor:
        output = None

        for i in range(input_tensor.size(0)):
            if hidden is not None:
                output, hidden = self.model(input_tensor[i], hidden)
            else:
                output = self.model(input_tensor[i])

        return criterion(output, label.unsqueeze(0))