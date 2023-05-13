# This will house the attention blocks that we will use in our models. We will start with the scaled dot product
# attention block.


# Attention is necessary because we want to pass information about previous words to the next word. We can do this by
# brute force using a for loop or more efficiently using matrix multiplication

from abc import ABC
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(ABC, nn.Module):
    # Abstract class for a head of an attention block

    def __init__(self, embedding_dim: int, output_dim: int, hard: Optional[bool] = False,
                 dropout_prob: Optional[float] = 0.0):
        """Initialize the attention block.
        Args:
            embedding_dim: The dimension of the embedding.
            output_dim: The dimension of the output. Embedding dim must be divisible by output dim.
            hard: Whether to use hard attention or not.
        """
        super(Attention, self).__init__()
        assert embedding_dim % output_dim == 0, f"The embedding dimension {embedding_dim} must be divisible by the " \
                                                f"output dimension {output_dim}."
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hard = hard
        self.dropout = nn.Dropout(dropout_prob)

    # @abstractmethod
    # def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Perform a forward pass of the attention block."""
    #     pass

    def apply_max(self, attention: torch.Tensor) -> torch.Tensor:
        """Apply the max operation to the attention tensor.
        Args:
            attention: The attention tensor.
        Returns:
            The attention tensor.
        """
        if self.hard:
            # Apply the max operation
            max_attention, max_indices = torch.max(attention, dim=-1)
            # Convert the max indices to a one-hot vector
            max_indices = F.one_hot(max_indices, attention.shape[-1]).float()
            # Convert the max indices to a one-hot vector
            attention = max_indices

        else:
            # Apply the softmax operation
            attention = F.softmax(attention, dim=-1)

        return attention


class SelfAttention(Attention):

    def __init__(self, embedding_dim: int, output_dim: int, hard: Optional[bool] = False,
                 dropout_prob: Optional[float] = 0.0):
        """Initialize the scaled dot product attention block.
        Args:
            embedding_dim: The dimension of the embedding.
            output_dim: The dimension of the output. Embedding dim must be divisible by output dim.
            hard: Whether to use hard attention or not.
            dropout_prob: The dropout probability.

        """
        super(SelfAttention, self).__init__(embedding_dim, output_dim, hard, dropout_prob)
        self.key = nn.Linear(embedding_dim, output_dim)
        self.query = nn.Linear(embedding_dim, output_dim)
        self.value = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of the self-attention block.
        Args:
            x: The input to the self-attention block.
        Returns:
            The output and attention tensors.
        """

        # Get input shape
        batch_size, seq_len, n_channels = x.shape

        # Get the keys, queries, and values
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Recall that attention is calculated as softmax(QK^T)V

        # Calculate QK^T
        attention = torch.matmul(queries, keys.transpose(1, 2)) * n_channels ** (-0.5)

        # Apply masking
        mask = torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0).to(x.device)
        attention = attention.masked_fill(mask == 0, float('-inf'))

        # Apply the max operation
        attention = self.apply_max(attention)

        # Calculate the output
        output = torch.matmul(attention, values)

        # Apply dropout
        output = self.dropout(output)

        return output, attention


class CrossAttention(Attention):

    def __init__(self, embedding_dim: int, output_dim: int, hard: Optional[bool] = False,
                 dropout_prob: Optional[float] = 0.0):
        """Initialize the scaled dot product attention block.
            Args:
                embedding_dim: The dimension of the embedding.
                output_dim: The dimension of the output. Embedding dim must be divisible by output dim.
                hard: Whether to use hard attention or not.
            """
        super(CrossAttention, self).__init__(embedding_dim, output_dim, hard)
        self.key = nn.Linear(embedding_dim, output_dim)
        self.query = nn.Linear(embedding_dim, output_dim)
        self.value = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of the cross-attention block.
            Args:
                x: The input to the cross-attention block.
                y: The input to the cross-attention block.
            Returns:
                The output and attention tensors.
            """

        # Get input shape
        batch_size, seq_len, n_channels = x.shape

        # Get the keys, queries, and values
        keys = self.key(x)
        queries = self.query(y)
        values = self.value(y)

        # Recall that attention is calculated as softmax(QK^T)V

        # Calculate QK^T
        attention = torch.matmul(queries, keys.transpose(1, 2)) * n_channels ** (-0.5)

        # Masking is not needed for cross-attention
        # mask = torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0).to(x.device)
        # attention = attention.masked_fill(mask == 0, float('-inf'))

        # Apply the max operation
        attention = self.apply_max(attention)

        # Calculate the output
        output = torch.matmul(attention, values)

        # Apply dropout
        output = self.dropout(output)

        return output, attention


class MultiHeadAttention(Attention):

    def __init__(self, embedding_dim: int, output_dim: int, n_heads: int, hard: Optional[bool] = False,
                 dropout_prob: Optional[float] = 0.0):
        """Initialize the multi-head self attention block.
        Args:
            embedding_dim: The dimension of the embedding.
            output_dim: The dimension of the output. Embedding dim must be divisible by output dim.
            n_heads: The number of heads to use.
            hard: Whether to use hard attention or not.
            dropout_prob: The dropout probability.
        """
        super(MultiHeadAttention, self).__init__(embedding_dim, output_dim, hard)
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.linear = nn.Linear(self.head_dim * n_heads, output_dim)
        self.heads = nn.ModuleList([SelfAttention(self.embedding_dim,
                                                  self.head_dim, self.hard) for _ in range(n_heads)])
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of the multi-head self attention block.
        Args:
            x: The input to the multi-head self attention block.
        Returns:
            The output and attention tensors."""

        # Apply the attention heads
        output, attention = torch.cat([head(x)[0] for head in self.heads], dim=-1), \
            torch.cat([head(x)[1] for head in self.heads], dim=-1)

        # Apply the linear layer
        output = self.linear(output)

        # Apply dropout
        output = self.dropout(output)

        return output, attention


class MultiHeadCrossAttention(Attention):
    """A multi-head cross attention block."""

    def __init__(self, embedding_dim: int, output_dim: int, n_heads: int, hard: Optional[bool] = False,
                 dropout_prob: Optional[float] = 0.0):
        """Initialize the multi-head cross attention block.
        Args:
            embedding_dim: The dimension of the embedding.
            output_dim: The dimension of the output. Embedding dim must be divisible by output dim.
            n_heads: The number of heads to use.
            hard: Whether to use hard attention or not.
            dropout_prob: The dropout probability.
        """
        super(MultiHeadCrossAttention, self).__init__(embedding_dim, output_dim, hard)
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.linear = nn.Linear(self.head_dim * n_heads, output_dim)
        self.heads = nn.ModuleList([CrossAttention(self.embedding_dim,
                                                   self.head_dim, self.hard) for _ in range(n_heads)])
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of the multi-head cross attention block.
        Args:
            x: The input to the multi-head cross attention block.
            y: The input to the multi-head cross attention block.
        Returns:
            The output and attention tensors."""

        # Apply the attention heads
        output, attention = torch.cat([head(x, y)[0] for head in self.heads], dim=-1), \
            torch.cat([head(x, y)[1] for head in self.heads], dim=-1)

        # Apply the linear layer
        output = self.linear(output)

        # Apply dropout
        output = self.dropout(output)

        return output, attention


class FeedForwardNetwork(nn.Module):
    """A feed forward network."""

    def __init__(self, embedding_dim: int, output_dim: int, hidden_dim: int, dropout_prob: Optional[float] = 0.0):
        """Initialize the feed forward network.
        Args:
            embedding_dim: The dimension of the embedding.
            output_dim: The dimension of the output. Embedding dim must be divisible by output dim.
            hidden_dim: The dimension of the hidden layer.
            dropout_prob: The dropout probability.
        """
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the feed forward network.
        Args:
            x: The input to the feed forward network.
        Returns:
            The output tensor.
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """A transformer block. This is used as encoder in the encoder-decoder architecture and as decoder in the decoder only architecture."""

    def __init__(self, embedding_dim: int, output_dim: int, num_heads: int, hard: Optional[bool] = False,
                 dropout_prob: Optional[float] = 0.0):
        """Initialize the transformer block. Here Masked Multi-Head Attention is used.
        Args:
            embedding_dim: The dimension of the embedding.
            output_dim: The dimension of the output. Embedding dim must be divisible by output dim.
            num_heads: The number of heads to use.
            hard: Whether to use hard attention or not.
            dropout_prob: The dropout probability.
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, output_dim, num_heads, hard, dropout_prob)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ff = FeedForwardNetwork(embedding_dim, output_dim, 4 * embedding_dim, dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the transformer block.
        Args:
            x: The input to the transformer block.
        Returns:
            The output tensor.
        """

        # First Apply the attention block
        attention, _ = self.attention(x)

        # Apply the first normalization (Add & Norm)
        x = self.norm1(x + attention)

        # Apply the feed forward network
        ff = self.ff(x)

        # Apply the second normalization (Add & Norm)
        x = self.norm2(x + ff)

        return x


class DecoderTransformerBlock(nn.Module):
    """A Fully Connected Decoder Transformer Block.

    This has both masked multi-head self-attention and multi-head cross-attention.
    """

    def __init__(self, embedding_dim: int, output_dim: int, num_heads: int, hard: Optional[bool] = False,
                 dropout_prob: Optional[float] = 0.0):
        """Initialize the transformer block. Here Masked Multi-Head Attention is used.
        Args:
            embedding_dim: The dimension of the embedding.
            output_dim: The dimension of the output. Embedding dim must be divisible by output dim.
            num_heads: The number of heads to use.
            hard: Whether to use hard attention or not.
            dropout_prob: The dropout probability.
        """
        super(DecoderTransformerBlock, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, output_dim, num_heads, hard, dropout_prob)
        self.cross_attention = MultiHeadCrossAttention(embedding_dim, output_dim, num_heads, hard, dropout_prob)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.ff = FeedForwardNetwork(embedding_dim, output_dim, 4 * embedding_dim, dropout_prob)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the transformer block.
        Args:
            x: The input to the transformer block.
        Returns:
            The output tensor.
        """

        # First Apply the self-attention block
        self_attention, _ = self.self_attention(x)

        # Apply the first normalization (Add & Norm)
        x = self.norm1(x + self_attention)

        # Apply the cross-attention block
        cross_attention, _ = self.cross_attention(x, y)

        # Apply the second normalization (Add & Norm)
        x = self.norm2(x + cross_attention)

        # Apply the feed forward network
        ff = self.ff(x)

        # Apply the third normalization (Add & Norm)
        x = self.norm3(x + ff)

        return x


if __name__ == '__main__':
    x = torch.rand(2, 10, 128)
    y = torch.rand(2, 10, 128)
    attention = MultiHeadAttention(128, 128, 8)
    output, attention = attention(x)
    print(output.shape)
    print(attention.shape)
    attention = CrossAttention(128, 128)
    output, attention = attention(x, y)
    print(output.shape)
    print(attention.shape)
    attention = SelfAttention(128, 128)
    output, attention = attention(x)
    print(output.shape)
    print(attention.shape)
