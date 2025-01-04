# %%
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import string


def generate_wrapped_char_data(n_samples=1000, seq_length=26):
    """
    Generate training data where one random character in a sequence is wrapped with <>.

    Args:
        n_samples (int): Number of sequences to generate
        seq_length (int): Length of each sequence (default 26 for A-Z)

    Returns:
        list: List of input sequences
        list: List of target characters (the wrapped characters)
    """
    sequences = []
    targets = []

    for _ in range(n_samples):
        # Generate a random permutation of A-Z
        chars = list(string.ascii_uppercase)
        random.shuffle(chars)

        # Choose a random position for the wrapped character
        wrap_pos = random.randint(0, seq_length - 1)
        target_char = chars[wrap_pos]

        # Create the sequence with wrapped character
        chars.insert(wrap_pos, "<")
        chars.insert(wrap_pos + 2, ">")
        sequence = "".join(chars)

        sequences.append(sequence)
        targets.append(target_char)

    vocab = list(string.ascii_uppercase) + ["<", ">"]

    return sequences, targets, vocab


def tokenize(sequences, vocab):
    retval = []
    for seq in sequences:
        r = []
        for char in seq:
            r.append(vocab.index(char))
        retval.append(r)
    return torch.tensor(retval)


sequences, targets, vocab = generate_wrapped_char_data(n_samples=1000)


sequences = tokenize(sequences, vocab)
targets = tokenize(targets, vocab)
dataset = TensorDataset(sequences, targets)

from torch.utils.data import Dataset

train_frac = 0.8
batch_size = 128

train_size = int(train_frac * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)

#  Define model
import pytorch_lightning as pyl


class CharDecoder(pyl.LightningModule):
    def __init__(self, vocab_size, output_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Embedding(vocab_size, vocab_size)

        # One-hot encoding
        self.embedding.weight.data = torch.eye(vocab_size)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        # x is a tensor of shape (batch_size, seq_len)
        batch_size, seq_len = x.shape

        # To token index to one-hot encoding
        x = self.embedding(x)

        # To sentnece to sequence of chars
        hidden = self.init_hidden(batch_size)
        x, _ = self.lstm(x, hidden)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y.reshape(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            y_hat = self(x)
            loss = torch.nn.functional.cross_entropy(y_hat, y.reshape(-1))
            self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=self.device),
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=self.device),
        )

model = CharDecoder(
    vocab_size=len(vocab),
    output_size=len(vocab),
    hidden_size=32
)

from pytorch_lightning import loggers as pl_loggers
tb_logger = pl_loggers.TensorBoardLogger('logs/')

trainer = pyl.Trainer(
    max_epochs=200,
    #enable_progress_bar=False,
    enable_model_summary=False,
    logger=tb_logger,
)
trainer.fit(model, train_dataloader, val_dataloader)
# %%
trainer.callback_metrics['val_loss']


eval_seq, eval_target, vocab = generate_wrapped_char_data(n_samples=5)
X_eval = tokenize(eval_seq, vocab)
y_eval = tokenize(eval_target, vocab)

model.eval()
with torch.no_grad():
    y_hat = model(X_eval)
    predicted_idx = torch.argmax(y_hat, dim=1)
    predicted_char = [vocab[idx] for idx in predicted_idx]

    for i in range(len(eval_seq)):
        print(f"Sequence: {eval_seq[i]}, Target: {eval_target[i]}, Predicted: {predicted_char[i]}")
    accuracy = (predicted_idx == y_eval).sum() / len(y_eval)
    print(f"Accuracy: {accuracy}")
# %%


def tokenize(sequences, vocab):
    retval = []
    for seq in sequences:
        r = []
        print(len(seq))
        for char in seq:
            r.append(vocab.index(char))
        retval.append(r)
    return torch.tensor(retval)

X = tokenize(['ABCDEFGHIJKLMNOPQRST<U>VWXYZ', 'ABCDEFGHIJKLMNOPQRSTU<V>WXYZ'], vocab)
print("X:", X)
print("Shape of X:", X.shape)
# %%
