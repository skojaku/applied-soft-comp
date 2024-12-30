# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from typing import List, Tuple, Optional, Union
import math
from asctools.rnn_trainer import RNNTrainer


class VigenereCipher:
    def __init__(self):
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        self.vocab_size = len(self.alphabet)

    def _prepare_text(self, text):
        """Remove non-alphabetic characters and convert to uppercase."""
        return "".join(
            char if char == " " else char
            for char in text
            if char in self.alphabet or char == " "
        )

    def _prepare_key(self, key, text_length):
        """Repeat the key to match the length of the text."""
        key = self._prepare_text(key)
        return (key * (text_length // len(key) + 1))[:text_length]

    def encrypt(self, plaintext, key):
        """
        Encrypt the plaintext using Vigenère cipher.

        Args:
            plaintext (str): The text to encrypt
            key (str): The encryption key

        Returns:
            str: The encrypted text
        """
        plaintext = self._prepare_text(plaintext)
        key = self._prepare_key(key, len(plaintext))
        ciphertext = ""

        for p, k in zip(plaintext, key):
            # Convert letters to numbers (A=0, B=1, etc.)
            p_idx = self.alphabet.index(p)
            k_idx = self.alphabet.index(k)

            # Apply Vigenère encryption formula
            c_idx = (
                p_idx + k_idx
            ) % self.vocab_size  # because we now have 26 letters + space + lowercase

            # Convert back to letter
            ciphertext += self.alphabet[c_idx]

        return ciphertext

    def decrypt(self, ciphertext, key):
        """
        Decrypt the ciphertext using Vigenère cipher.

        Args:
            ciphertext (str): The text to decrypt
            key (str): The decryption key

        Returns:
            str: The decrypted text
        """
        ciphertext = self._prepare_text(ciphertext)
        key = self._prepare_key(key, len(ciphertext))
        plaintext = ""

        for c, k in zip(ciphertext, key):
            # Convert letters to numbers (A=0, B=1, etc.)
            c_idx = self.alphabet.index(c)
            k_idx = self.alphabet.index(k)

            # Apply Vigenère decryption formula
            p_idx = (
                c_idx - k_idx
            ) % self.vocab_size  # 27 because we now have 26 letters + space

            # Convert back to letter
            plaintext += self.alphabet[p_idx]

        return plaintext


cipher = VigenereCipher()

# Test the cipher
message = "Hello World"
key = "SECRET"

# Encryption
encrypted = cipher.encrypt(message, key)
print(f"Original message: {message}")
print(f"Key: {key}")
print(f"Encrypted message: {encrypted}")

# Decryption
decrypted = cipher.decrypt(encrypted, key)
print(f"Decrypted message: {decrypted}")

seq_len = 5
n_samples = 1000
key = "PASSWORD"
target_seq, input_seq = [], []
for _ in range(n_samples):
    target_seq.append(
        "".join(
            np.random.choice(
                list("ABCDEFGHIJKLMNOPQRSTUVWXYZ "), size=seq_len, replace=True
            )
        )
    )
    input_seq.append(cipher.encrypt(target_seq[-1], key))


def to_one_hot(sequences, seq_len, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
    one_hot = torch.zeros((len(sequences), seq_len, len(alphabet)))
    for i in range(len(sequences)):
        for j in range(seq_len):
            one_hot[i, j, alphabet.index(sequences[i][j])] = 1
    return one_hot


input_tensor = to_one_hot(input_seq, seq_len)
target_tensor = to_one_hot(target_seq, seq_len)


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]

        batch_size, seq_len, hidden_size = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Calculate attention scores
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch_size, 1, hidden_size]
        attention_weights = F.softmax(torch.bmm(v, energy), dim=2)  # [batch_size, 1, seq_len]

        # Apply attention to encoder outputs
        context = torch.bmm(attention_weights, encoder_outputs)  # [batch_size, 1, hidden_size]

        return context, attention_weights

class EncoderDecoderLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 27,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: str = "cpu" if not torch.cuda.is_available() else "cuda",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = AttentionLayer(hidden_size)

        # Additional layers
        self.combine_context = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size)
        )

        self.device = device

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_outputs: Optional[torch.Tensor] = None,
        mode: str = 'encode'
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],  # For encode mode
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]  # For decode mode
    ]:
        if mode == 'encode':
            # Encode sequence
            encoder_outputs, encoder_hidden = self.encoder(x, hidden)
            return encoder_outputs, encoder_hidden

        elif mode == 'decode':
            if encoder_outputs is None:
                raise ValueError("encoder_outputs is required for decode mode")

            # Run decoder step
            decoder_output, decoder_hidden = self.decoder(x, hidden)

            # Calculate attention
            context, attention_weights = self.attention(decoder_hidden[0][-1], encoder_outputs)

            # Combine decoder output with context
            combined = torch.cat([decoder_output, context], dim=2)
            combined = self.combine_context(combined)

            # Generate output
            output = self.output(combined)
            output = F.log_softmax(output, dim=-1)

            return output, decoder_hidden, attention_weights

        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'encode' or 'decode'")

    def initHidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=self.device
            ),
            torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=self.device
            ),
        )

    def generate(self, input_seq: torch.Tensor, seq_len: int = 20):
        with torch.no_grad():
            # Initial encoder pass
            encoder_outputs, encoder_hidden = self.forward(input_seq, mode='encode')

            # Initialize decoder
            decoder_hidden = encoder_hidden
            decoder_input = input_seq[:, :1, :]  # Start with first input token

            generated_seq = []
            for t in range(seq_len):
                # Decode step
                output, decoder_hidden, _ = self.forward(
                    decoder_input,
                    decoder_hidden,
                    encoder_outputs=encoder_outputs,
                    mode='decode'
                )

                # Sample from the output distribution instead of taking argmax
                probs = torch.exp(output.squeeze(1))
                pred = torch.multinomial(probs, 1)
                decoder_input = F.one_hot(pred, num_classes=self.input_size).float()

                generated_seq.append(pred.item())

            return generated_seq

# Initialize and train the model
model = EncoderDecoderLSTM()
trainer = RNNTrainer(model)
losses = trainer.train(
    input_tensors=input_tensor,
    targets=target_tensor,
    criterion=nn.CrossEntropyLoss(),
    max_epochs=1000,
    learning_rate=0.001,
    batch_size=64,
    patience=20,
    clip_grad_norm=1.0,
)

# %%
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()
# %%
model.eval()
eval_seq = input_seq[0]
enc_seq = cipher.encrypt(eval_seq, key)
eval_tensor = to_one_hot([enc_seq], seq_len=len(eval_seq))

print(eval_seq)
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
print(''.join([alphabet[i] for i in model.generate(eval_tensor, seq_len=len(eval_seq))]))
# %%
