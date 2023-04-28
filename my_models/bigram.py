import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from my_models.AbstractModelClass import AbstractModelClass


class BigramModel(AbstractModelClass):
    def __init__(self, vocab_size: int, embedding_dim: int):
        """Initialize the Bigram model by setting up the various layers."""
        super(BigramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # The Token Embedding Table

    def forward(self, idx: torch.Tensor, target: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of our model on some input and target text."""
        embeds = self.embeddings(idx)

        if target is None:
            # If no target is provided, no loss is provided to avoid an error
            loss = None
        else:
            B, T, C = embeds.shape  # Batch by Time by Channels
            embeds = embeds.view(B * T, C) # Need to reshape the input so that the Channel dimension is second (2D array)
            target = target.view(B * T) # Need to reshape the target so that it is a 1D array
            loss = F.cross_entropy(embeds, target)
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
            idx = torch.cat((idx, idx_next), dim=1) # Shapes are important here (B
        return idx
