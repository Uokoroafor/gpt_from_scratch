# A file to create the abstract class for the models to inherit from

from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
import torch.nn as nn


class AbstractModelClass(ABC, nn.Module):

    def __init__(self):
        """Initialize the model by setting up the various layers."""
        super(AbstractModelClass, self).__init__()

    @abstractmethod
    def forward(self, idx: torch.Tensor, target: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of our model on some input and target text."""
        pass

    @abstractmethod
    def generate(self, idx: torch.Tensor, length: int) -> List[int]:
        """Generate text using the model.
        Args:
            idx: The initial index of the text to generate.
            length: The length of the text to generate.
        Returns:
            A list of integers representing the generated text.
        """
        pass

    def save(self, path: str):
        """Save the model to a file."""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load the model from a file."""
        self.load_state_dict(torch.load(path))