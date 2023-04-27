import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class BigramModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        """Initialize the Bigram model by setting up the various layers."""
        super(BigramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, idx: torch.Tensor, target: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of our model on some input and target text."""
        embeds = self.embeddings(idx)

        if target is None:
            loss = None
        else:
            B, T, C = embeds.shape
            embeds = embeds.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(embeds, target)
            # print('loss', loss)
            # print('embeds', embeds.shape)

        return embeds, loss

    def generate(self, idx: torch.Tensor, length: int) -> List[int]:
        """Generate text using the model."""
        for _ in range(length):
            # Forward pass
            embeds, loss = self.forward(idx)


            # Focus only on the last word
            embeds = embeds[:, -1, :]
            # Get the index of the word (by sampling using softmax)
            idx_next = F.softmax(embeds, dim=1)

            # Sample over these probabilities
            idx_next = torch.multinomial(idx_next, 1)

            # Add the index to the list of indices
            idx = torch.cat((idx, idx_next), dim=1)
        return idx






