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

    def test_many_to_many(self):
        """Test sequence-to-sequence prediction with same length input/output"""
        seq_length = 5
        batch_size = 2

        # Create input and target sequences
        input_tensor = torch.randn(batch_size, seq_length, self.input_size).to(self.device)
        target_tensor = torch.randn(batch_size, seq_length, self.output_size).to(self.device)

        # Train model
        losses = self.trainer.train(
            input_tensors=input_tensor,
            targets=target_tensor,
            criterion=nn.MSELoss(),
            max_epochs=5,
            learning_rate=0.01
        )

        self.assertTrue(len(losses) > 0)
        self.assertIsInstance(losses[0], float)
        self.assertTrue(all(isinstance(loss, float) for loss in losses))

    def test_many_to_one(self):
        """Test sequence to single output prediction"""
        seq_length = 5
        batch_size = 2

        # Create input sequence and single target
        input_tensor = torch.randn(batch_size, seq_length, self.input_size).to(self.device)
        target_tensor = torch.randn(batch_size, 1, self.output_size).to(self.device)

        # Train model
        losses = self.trainer.train(
            input_tensors=input_tensor,
            targets=target_tensor,
            criterion=nn.MSELoss(),
            max_epochs=5,
            learning_rate=0.01
        )

        self.assertTrue(len(losses) > 0)
        self.assertIsInstance(losses[0], float)
        self.assertTrue(all(isinstance(loss, float) for loss in losses))

    def test_classification(self):
        """Test sequence classification (special case of many-to-one)"""
        seq_length = 5
        batch_size = 2
        num_classes = self.output_size

        # Create input sequences
        input_tensor = torch.randn(batch_size, seq_length, self.input_size).to(self.device)

        # Create one-hot encoded target classes
        target_tensor = torch.zeros(batch_size, 1, num_classes).to(self.device)
        for i in range(batch_size):
            target_class = torch.randint(0, num_classes, (1,))
            target_tensor[i, 0, target_class] = 1

        # Train model
        losses = self.trainer.train(
            input_tensors=input_tensor,
            targets=target_tensor,
            criterion=nn.BCEWithLogitsLoss(),  # or CrossEntropyLoss with proper reshaping
            max_epochs=5,
            learning_rate=0.01
        )

        self.assertTrue(len(losses) > 0)
        self.assertIsInstance(losses[0], float)
        self.assertTrue(all(isinstance(loss, float) for loss in losses))

    def test_batch_of_sequences(self):
        """Test training with multiple sequences"""
        seq_length = 5
        batch_size = 2
        num_sequences = 3

        # Create list of input sequences and their targets
        input_tensors = [
            torch.randn(batch_size, seq_length, self.input_size).to(self.device)
            for _ in range(num_sequences)
        ]
        target_tensor = torch.randn(batch_size, 1, self.output_size).to(self.device)

        # Train model
        losses = self.trainer.train(
            input_tensors=input_tensors,
            targets=target_tensor,
            criterion=nn.MSELoss(),
            max_epochs=5,
            learning_rate=0.01
        )

        self.assertTrue(len(losses) > 0)
        self.assertIsInstance(losses[0], float)
        self.assertTrue(all(isinstance(loss, float) for loss in losses))

    def test_invalid_target_shape(self):
        """Test that invalid target shapes raise errors"""
        seq_length = 5
        batch_size = 2

        input_tensor = torch.randn(batch_size, seq_length, self.input_size)
        invalid_target = torch.randn(batch_size, seq_length)  # Missing feature dimension

        with self.assertRaises(RuntimeError):
            self.trainer.train(
                input_tensors=input_tensor,
                targets=invalid_target,
                max_epochs=1
            )

    def test_multiclass_classification(self):
        """Test multi-class sequence classification"""
        seq_length = 5
        batch_size = 4
        num_classes = 10  # More classes to test multi-class scenario

        # Create input sequences
        input_tensor = torch.randn(batch_size, seq_length, self.input_size).to(self.device)

        # Create target classes (both one-hot and class index versions for different loss functions)
        # One-hot encoded targets
        onehot_targets = torch.zeros(batch_size, 1, num_classes).to(self.device)
        class_indices = torch.randint(0, num_classes, (batch_size,))
        for i in range(batch_size):
            onehot_targets[i, 0, class_indices[i]] = 1

        # Test with BCEWithLogitsLoss (one-hot targets)
        losses_bce = self.trainer.train(
            input_tensors=input_tensor,
            targets=onehot_targets,
            criterion=nn.BCEWithLogitsLoss(),
            max_epochs=5,
            learning_rate=0.01
        )

        self.assertTrue(len(losses_bce) > 0)
        self.assertIsInstance(losses_bce[0], float)

        # Test with CrossEntropyLoss (class indices)
        # Reshape targets for CrossEntropyLoss: (batch_size, 1, num_classes) -> (batch_size, num_classes)
        index_targets = class_indices.unsqueeze(1).to(self.device)
        losses_ce = self.trainer.train(
            input_tensors=input_tensor,
            targets=index_targets,
            criterion=nn.CrossEntropyLoss(),
            max_epochs=5,
            learning_rate=0.01
        )

        self.assertTrue(len(losses_ce) > 0)
        self.assertIsInstance(losses_ce[0], float)

        # Verify model outputs proper shape
        with torch.no_grad():
            hidden = self.model.initHidden(batch_size)
            output, _ = self.model(input_tensor[:, 0], hidden)  # Test first timestep
            self.assertEqual(output.shape, (batch_size, num_classes))

    def test_next_token_prediction(self):
        """Test next token prediction (predict only the next token)"""
        seq_length = 5
        batch_size = 2
        vocab_size = self.output_size

        # Create discrete input sequence (indices)
        input_indices = torch.randint(0, vocab_size, (batch_size, seq_length)).to(self.device)
        # Target is the next token for each sequence
        target_indices = torch.randint(0, vocab_size, (batch_size, 1)).to(self.device)

        # Convert to one-hot encoded tensors
        input_tensor = torch.zeros(batch_size, seq_length, vocab_size).to(self.device)
        target_tensor = torch.zeros(batch_size, 1, vocab_size).to(self.device)

        # Create one-hot encodings
        for b in range(batch_size):
            for t in range(seq_length):
                input_tensor[b, t, input_indices[b, t]] = 1
            target_tensor[b, 0, target_indices[b]] = 1

        # Train model
        losses = self.trainer.train(
            input_tensors=input_tensor,
            targets=target_tensor,
            criterion=nn.CrossEntropyLoss(),
            max_epochs=5,
            learning_rate=0.01
        )

        self.assertTrue(len(losses) > 0)
        self.assertIsInstance(losses[0], float)

        # Test prediction
        with torch.no_grad():
            # Process a sequence and predict the next token
            hidden = self.model.initHidden(batch_size)

            # Process the input sequence
            for t in range(seq_length):
                output, hidden = self.model(input_tensor[:, t], hidden)

            # Output should be probabilities over vocabulary
            self.assertEqual(output.shape, (batch_size, vocab_size))

            # Convert to discrete token
            predicted_token = torch.zeros_like(output)
            predicted_token.scatter_(1, output.argmax(1, keepdim=True), 1)

            # Verify it's a valid one-hot vector
            self.assertTrue(torch.all(predicted_token.sum(dim=1) == 1))
            self.assertTrue(torch.all((predicted_token == 0) | (predicted_token == 1)))

if __name__ == '__main__':
    unittest.main()
