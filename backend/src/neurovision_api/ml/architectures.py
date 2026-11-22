"""Alternative neural network architectures for MNIST."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTConv(nn.Module):
    """Convolutional Neural Network for MNIST.
    
    Architecture:
    - Conv2d(1, 32, 5) + ReLU + MaxPool2d(2)
    - Conv2d(32, 64, 5) + ReLU + MaxPool2d(2)
    - Flatten
    - Linear(1024, 128) + ReLU
    - Dropout(0.5)
    - Linear(128, 10)
    
    Parameters: ~50K
    Expected accuracy: 99.2%+
    """

    def __init__(self, dropout_p: float = 0.5) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, 10)
        """
        # First convolutional block
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second convolutional block
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x

    def get_activations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get intermediate activations for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        activations = {}
        
        x = self.conv1(x)
        activations["conv1"] = x.clone()
        x = F.relu(x)
        activations["conv1_relu"] = x.clone()
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.conv2(x)
        activations["conv2"] = x.clone()
        x = F.relu(x)
        activations["conv2_relu"] = x.clone()
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        activations["fc1"] = x.clone()
        x = F.relu(x)
        activations["fc1_relu"] = x.clone()
        
        x = self.fc2(x)
        activations["output"] = x.clone()
        
        return activations


class MNISTRnn(nn.Module):
    """Recurrent Neural Network for MNIST.
    
    Treats each row of the 28x28 image as a 28-step sequence.
    
    Architecture:
    - LSTM(28, 64, 2 layers, bidirectional)
    - Linear(128, 10)
    
    Parameters: ~38K
    Expected accuracy: 97.5%+
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, bidirectional: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.rnn = nn.LSTM(
            input_size=28,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(rnn_output_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RNN.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, 10)
        """
        # Reshape from (batch, 1, 28, 28) to (batch, 28, 28)
        # Treat height as sequence length and width as features
        batch_size = x.size(0)
        x = x.squeeze(1)  # (batch, 28, 28)
        
        # LSTM forward
        _, (h_n, _) = self.rnn(x)
        
        # Take last hidden state from all layers
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # We'll use the last layer's hidden state
        if self.bidirectional:
            h_n = h_n[-2:, :, :]  # Last 2 (forward and backward)
            h_n = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, 2*hidden_size)
        else:
            h_n = h_n[-1, :, :]  # (batch, hidden_size)
        
        # Fully connected layer
        output = self.fc(h_n)
        
        return output


class MNISTTransformer(nn.Module):
    """Transformer model for MNIST.
    
    Treats the 28x28 image as 784 tokens (one per pixel).
    
    Architecture:
    - Positional Encoding
    - TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, num_layers=4)
    - Linear(128, 10)
    
    Parameters: ~230K
    Expected accuracy: 98.5%+
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        
        # Embedding layer: project each pixel value to d_model dimensions
        self.embedding = nn.Linear(1, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            self._create_positional_encoding(784, d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Linear(d_model, 10)

    @staticmethod
    def _create_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding using sine and cosine."""
        import math
        
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Transformer.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, 10)
        """
        batch_size = x.size(0)
        
        # Flatten and reshape: (batch, 1, 28, 28) -> (batch, 784, 1)
        x = x.view(batch_size, -1, 1)
        
        # Embed pixel values: (batch, 784, 1) -> (batch, 784, d_model)
        x = self.embedding(x)
        
        # Add positional encoding
        x = x + self.positional_encoding.unsqueeze(0)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling: (batch, 784, d_model) -> (batch, d_model)
        x = x.mean(dim=1)
        
        # Classification head: (batch, d_model) -> (batch, 10)
        output = self.fc(x)
        
        return output
