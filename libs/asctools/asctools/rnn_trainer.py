import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple, Callable
from torch.optim import Optimizer
from torch.nn import Module
from tqdm import tqdm

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
        patience: int = 30,
        min_delta: float = 1e-4,
        hidden_init_func: Optional[Callable] = None,
        clip_grad_norm: Optional[float] = 1.0,
        lr_scheduler: Optional[str] = 'reduce_on_plateau',
        lr_patience: int = 5,
        lr_factor: float = 0.5
    ) -> List[float]:
        """
        Train the sequence model

        Args:
            input_tensors: Input sequence tensor(s) of shape (batch_size, seq_length, feature_size)
            task: 'next_token', 'classification', or 'sequence'
            labels: Class labels for classification task
            targets: Targets for next token prediction or sequence prediction
            criterion: Loss function
            optimizer: Optimizer
            max_epochs: Maximum number of training epochs
            learning_rate: Learning rate for optimizer
            teacher_forcing_ratio: Probability of using teacher forcing
            patience: Number of epochs to wait for improvement before early stopping
            min_delta: Minimum change in loss to qualify as an improvement
            hidden_init_func: Function to initialize hidden state
            lr_scheduler: Learning rate scheduler type
            lr_patience: Number of epochs to wait for improvement before reducing learning rate
            lr_factor: Factor by which to reduce learning rate

        Returns:
            losses: List of losses during training
        """
        # Ensure inputs are on the correct device
        if isinstance(input_tensors, torch.Tensor):
            input_tensors = input_tensors.to(self.device)
        else:
            input_tensors = [t.to(self.device) for t in input_tensors]

        if targets is not None:
            targets = targets.to(self.device)

        criterion = criterion or (nn.MSELoss() if task == 'sequence' else nn.CrossEntropyLoss())
        optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Add learning rate scheduler
        scheduler = None
        if lr_scheduler == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=lr_factor, patience=lr_patience, verbose=True
            )

        losses = []
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):
            epoch_loss = 0

            # Handle both single tensor and list of tensors
            if isinstance(input_tensors, torch.Tensor):
                input_list = [input_tensors]
                target_list = [targets] if targets is not None else [None]
            else:
                input_list = input_tensors
                target_list = targets if targets is not None else [None] * len(input_list)

            for input_tensor, target in zip(input_list, target_list):
                hidden = hidden_init_func() if hidden_init_func else None
                optimizer.zero_grad()

                if task == 'sequence':
                    sequence_loss = self._train_sequence(
                        input_tensor,
                        target,
                        hidden,
                        criterion,
                        teacher_forcing_ratio
                    )
                elif task == 'next_token':
                    sequence_loss = self._train_next_token(
                        input_tensor,
                        target,
                        hidden,
                        criterion,
                        teacher_forcing_ratio
                    )
                else:  # Classification task
                    sequence_loss = self._train_classification(
                        input_tensor, hidden, criterion, labels
                    )

                sequence_loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                optimizer.step()
                epoch_loss += sequence_loss.item()

            avg_loss = epoch_loss / len(input_list)
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

            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step(avg_loss)

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

    def _train_sequence(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        hidden: Optional[torch.Tensor],
        criterion: Module,
        teacher_forcing_ratio: float
    ) -> torch.Tensor:
        sequence_loss = 0
        output = None

        # Get dimensions
        batch_size = input_tensor.size(0)
        seq_length = input_tensor.size(1) if input_tensor.dim() > 1 else 1

        # Reshape tensors if needed
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(-1)
        if target_tensor.dim() == 2:
            target_tensor = target_tensor.unsqueeze(-1)

        # Initialize hidden state
        if hidden is None:
            hidden = self.model.initHidden(batch_size)

        # Iterate through sequence
        for i in range(seq_length - 1):
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

            if use_teacher_forcing or i == 0:
                curr_input = input_tensor[:, i, :]
            else:
                curr_input = output

            output, hidden = self.model(curr_input, hidden)
            target = target_tensor[:, i, :]
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