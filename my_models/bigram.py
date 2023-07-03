from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_models.AbstractModelClass import AbstractModelClass
from my_models.attention_block import MultiHeadAttention, TransformerBlock


class BigramModel(AbstractModelClass):
    def __init__(self, vocab_size: int, embedding_dim: int):
        """Initialize the Bigram model by setting up the various layers."""
        super(BigramModel, self).__init__()
        self.embeddings = nn.Embedding(
            vocab_size, embedding_dim
        )  # The Token Embedding Table

    def forward(
        self, src: torch.Tensor, trg: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of our model on some input and target text."""
        embeds = self.embeddings(src)

        if trg is None:
            # If no target is provided, no loss is provided to avoid an error
            loss = None
        else:
            B, T, C = embeds.shape  # Batch by Time by Channels
            embeds = embeds.view(
                B * T, C
            )  # Need to reshape the input so that the Channel dimension is second (2D array)
            trg = trg.view(B * T)  # Need to reshape the target so that it is a 1D array
            loss = F.cross_entropy(embeds, trg)
            # How well are we predicting the next character based on the current character?

        return embeds, loss

    def generate(self, idx: torch.Tensor, length: int) -> List[int]:
        """Generate text using the model.
        Args:
            idx: The initial index of the text to generate.
            length: The length of the text to generate.
        Returns:
            A list of integers representing the generated text.

        """
        for _ in range(length):
            # Make a prediction from the current index (Forward pass)
            embeds, loss = self.forward(idx)

            # Focus only on the last word
            embeds = embeds[:, -1, :]

            # Get the index of the word (by sampling using softmax)
            idx_next = F.softmax(embeds, dim=1)

            # Sample over these probabilities
            idx_next = torch.multinomial(idx_next, 1)

            # Add the index to the list of indices
            idx = torch.cat((idx, idx_next), dim=1)  # Shapes are important here (B
        return idx


class BigramModelWithAttention(BigramModel):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        block_size: int,
        num_heads: Optional[int] = 4,
    ):
        """Initialize the Bigram model by setting up the various layers.
        Args:

        """
        super(BigramModelWithAttention, self).__init__(vocab_size, embedding_dim)
        # self.attention = SelfAttention(embedding_dim, embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, embedding_dim, num_heads)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.block_size = block_size

    def forward(
        self, src: torch.Tensor, trg: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of our model on some input and target text."""
        embeds = self.embeddings(src)
        embeds, _ = self.attention(embeds)
        embeds = self.linear(embeds)

        if trg is None:
            # If no target is provided, no loss is provided to avoid an error
            loss = None
        else:
            B, T, C = embeds.shape
            embeds = embeds.view(B * T, C)
            trg = trg.view(B * T)
            loss = F.cross_entropy(embeds, trg)

        return embeds, loss

    def generate(self, idx: torch.Tensor, length: int) -> List[int]:
        """Generate text using the model.
        Args:
            idx: The initial index of the text to generate.
            length: The length of the text to generate.
        Returns:
            A list of integers representing the generated text.

        """
        for _ in range(length):
            # Make a prediction from the current index (Forward pass)
            # Now feed the block into the model
            idx_block = idx[:, -self.block_size :]

            embeds, loss = self.forward(idx_block)

            # Focus only on the last word
            embeds = embeds[:, -1, :]

            # Get the index of the word (by sampling using softmax)
            idx_next = F.softmax(embeds, dim=-1)

            # Sample over these probabilities
            idx_next = torch.multinomial(idx_next, 1)

            # Add the index to the list of indices
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class BigramModelWithAandPE(BigramModelWithAttention):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        block_size: int,
        num_heads: Optional[int] = 4,
    ):
        """Initialize the Bigram model by setting up the various layers."""
        super(BigramModelWithAandPE, self).__init__(
            vocab_size, embedding_dim, block_size, num_heads
        )
        self.positional_encoding = nn.Embedding(block_size, embedding_dim)

    def forward(
        self, src: torch.Tensor, trg: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of our model on some input and target text."""
        B, T = src.shape
        embeds = self.embeddings(src)
        position = torch.arange(T, device=embeds.device)  # .expand(B, T)
        embeds = embeds + self.positional_encoding(position)
        embeds, _ = self.attention(embeds)
        embeds = self.linear(embeds)

        if trg is None:
            # If no target is provided, no loss is provided to avoid an error
            loss = None
        else:
            B, T, C = embeds.shape
            embeds = embeds.view(B * T, C)
            trg = trg.view(B * T)
            loss = F.cross_entropy(embeds, trg)

        return embeds, loss


class BigramModelWithAandPEandLN(BigramModelWithAandPE):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        block_size: int,
        num_heads: Optional[int] = 4,
    ):
        """Initialize the Bigram model by setting up the various layers."""
        super(BigramModelWithAandPEandLN, self).__init__(
            vocab_size, embedding_dim, block_size, num_heads
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, src: torch.Tensor, trg: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of our model on some input and target text."""
        B, T = src.shape
        embeds = self.embeddings(src)
        position = torch.arange(T, device=embeds.device)  # .expand(B, T)
        embeds = embeds + self.positional_encoding(position)
        embeds, _ = self.attention(embeds)
        embeds = self.layer_norm(embeds)
        embeds = self.linear(embeds)

        if trg is None:
            # If no target is provided, no loss is provided to avoid an error
            loss = None
        else:
            B, T, C = embeds.shape
            embeds = embeds.view(B * T, C)
            trg = trg.view(B * T)
            loss = F.cross_entropy(embeds, trg)

        return embeds, loss


class BigramModelWithAandPEandLNandFFN(BigramModelWithAandPEandLN):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        block_size: int,
        num_heads: Optional[int] = 4,
    ):
        """Initialize the Bigram model by setting up the various layers."""
        super(BigramModelWithAandPEandLNandFFN, self).__init__(
            vocab_size, embedding_dim, block_size, num_heads
        )
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(
        self, src: torch.Tensor, trg: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of our model on some input and target text."""
        B, T = src.shape
        embeds = self.embeddings(src)
        position = torch.arange(T, device=embeds.device)
        embeds = embeds + self.positional_encoding(position)
        embeds, _ = self.attention(embeds)
        embeds = self.layer_norm(embeds)
        embeds = self.ffn(embeds)
        embeds = self.linear(embeds)

        if trg is None:
            # If no target is provided, no loss is provided to avoid an error
            loss = None
        else:
            B, T, C = embeds.shape
            embeds = embeds.view(B * T, C)
            trg = trg.view(B * T)
            loss = F.cross_entropy(embeds, trg)

        return embeds, loss


class BigramModelWithAandPEandLNandFFNandDO(BigramModelWithAandPEandLNandFFN):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        block_size: int,
        num_heads: Optional[int] = 4,
        dropout_prob: Optional[float] = 0.0,
    ):
        """Initialize the Bigram model by setting up the various layers.
        Args:
            vocab_size: The size of the vocabulary.
            embedding_dim: The dimension of the embedding.
            block_size: The size of the blocks.
            num_heads: The number of heads to use in the attention layer.
            dropout_prob: The probability of dropout.
        """
        super(BigramModelWithAandPEandLNandFFNandDO, self).__init__(
            vocab_size, embedding_dim, block_size, num_heads
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.attention = MultiHeadAttention(
            embedding_dim, embedding_dim, num_heads, dropout_prob
        )

    def forward(
        self, src: torch.Tensor, trg: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of our model on some input and target text."""
        B, T = src.shape
        embeds = self.embeddings(src)
        position = torch.arange(T, device=embeds.device)
        embeds = embeds + self.positional_encoding(position)
        embeds, _ = self.attention(embeds)
        embeds = self.layer_norm(embeds)
        embeds = self.ffn(embeds)
        embeds = self.dropout(embeds)
        embeds = self.linear(embeds)

        if trg is None:
            # If no target is provided, no loss is provided to avoid an error
            loss = None
        else:
            B, T, C = embeds.shape
            embeds = embeds.view(B * T, C)
            trg = trg.view(B * T)
            loss = F.cross_entropy(embeds, trg)

        return embeds, loss


class BigramWithTransformerBlocks(BigramModelWithAandPEandLNandFFNandDO):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        block_size: int,
        num_heads: Optional[int] = 4,
        dropout_prob: Optional[float] = 0.0,
        num_blocks: Optional[int] = 1,
    ):
        """Initialize the Bigram model by setting up the various layers.
        Args:
            vocab_size: The size of the vocabulary.
            embedding_dim: The dimension of the embedding.
            block_size: The size of the blocks.
            num_heads: The number of heads to use in the attention layer.
            dropout_prob: The probability of dropout.
            num_blocks: The number of transformer blocks to use.
        """
        super(BigramWithTransformerBlocks, self).__init__(
            vocab_size, embedding_dim, block_size, num_heads, dropout_prob
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    output_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self, src: torch.Tensor, trg: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of our model on some input and target text."""
        B, T = src.shape
        embeds = self.embeddings(src)
        position = torch.arange(T, device=embeds.device)
        embeds = embeds + self.positional_encoding(position)
        for block in self.blocks:
            embeds = block(embeds)

        # Perform a final layer normalization and linear transformation
        embeds = self.layer_norm(embeds)
        embeds = self.linear(embeds)

        if trg is None:
            # If no target is provided, no loss is provided to avoid an error
            loss = None
        else:
            B, T, C = embeds.shape
            embeds = embeds.view(B * T, C)
            trg = trg.view(B * T)
            loss = F.cross_entropy(embeds, trg)

        return embeds, loss
