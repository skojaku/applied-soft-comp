import unittest
import torch
import torch.nn as nn
from asctools.rnn import RNN
from asctools.rnn_trainer import RNNTrainer

class TestRNN(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size = 20
        self.output_size = 10
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize model
        self.model = RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            device=self.device
        )

        # Initialize trainer
        self.trainer = RNNTrainer(self.model, device=self.device)

    def test_rnn_forward(self):
        # Test single forward pass
        batch_size = 1
        input_tensor = torch.randn(batch_size, self.input_size).to(self.device)
        hidden = self.model.initHidden()

        output, new_hidden = self.model(input_tensor, hidden)

        self.assertEqual(output.shape, (batch_size, self.output_size))
        self.assertEqual(new_hidden.shape, (batch_size, self.hidden_size))
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(new_hidden, torch.Tensor)

    def test_next_token_prediction(self):
        # Create sample sequence data
        seq_length = 5
        batch_size = 1

        # Create input sequence and targets
        input_tensor = torch.randint(0, self.input_size, (seq_length, batch_size))
        targets = torch.randint(0, self.output_size, (seq_length, batch_size))

        # Convert to one-hot encoding
        one_hot = torch.zeros(seq_length, batch_size, self.input_size)
        for i in range(seq_length):
            one_hot[i, 0, input_tensor[i]] = 1

        # Train model
        losses = self.trainer.train(
            input_tensors=one_hot,
            task='next_token',
            targets=targets,
            max_epochs=10,
            learning_rate=0.01,
            teacher_forcing_ratio=0.5,
            hidden_init_func=self.model.initHidden
        )

        self.assertTrue(len(losses) > 0)
        self.assertIsInstance(losses[0], float)
        self.assertTrue(all(isinstance(loss, float) for loss in losses))

    def test_classification(self):
        # Create sample sequence data with labels
        seq_length = 5
        batch_size = 1
        num_samples = 3

        input_tensors = []
        labels = torch.zeros(num_samples, dtype=torch.long)

        for i in range(num_samples):
            sequence = torch.randn(seq_length, batch_size, self.input_size)
            input_tensors.append(sequence)
            labels[i] = i % 2  # Binary classification labels

        # Train model
        losses = self.trainer.train(
            input_tensors=input_tensors,
            task='classification',
            labels=labels,
            max_epochs=10,
            learning_rate=0.01,
            hidden_init_func=self.model.initHidden
        )

        self.assertTrue(len(losses) > 0)
        self.assertIsInstance(losses[0], float)
        self.assertTrue(all(isinstance(loss, float) for loss in losses))

    def test_invalid_task(self):
        # Test invalid task raises ValueError
        with self.assertRaises(ValueError):
            self.trainer.train(
                input_tensors=[torch.randn(5, 1, self.input_size)],
                task='invalid_task',
                max_epochs=1
            )

    def test_missing_targets(self):
        # Test missing targets for next_token task raises ValueError
        with self.assertRaises(ValueError):
            self.trainer.train(
                input_tensors=[torch.randn(5, 1, self.input_size)],
                task='next_token',
                max_epochs=1
            )

    def test_missing_labels(self):
        # Test missing labels for classification task raises ValueError
        with self.assertRaises(ValueError):
            self.trainer.train(
                input_tensors=[torch.randn(5, 1, self.input_size)],
                task='classification',
                max_epochs=1
            )

if __name__ == '__main__':
    unittest.main()
