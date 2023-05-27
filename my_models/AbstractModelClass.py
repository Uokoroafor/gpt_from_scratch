# A file to create the abstract class for the models to inherit from
import os.path
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple, Optional
import torch
import torch.nn as nn


class AbstractModelClass(ABC, nn.Module):

    def __init__(self):
        """Initialize the model by setting up the various layers."""
        super(AbstractModelClass, self).__init__()

    @abstractmethod
    def forward(self, src: torch.Tensor, trg: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of our model on some input and target text."""
        pass

    # @abstractmethod
    # def generate(self, idx: torch.Tensor, length: int) -> List[int]:
    #     """Generate text using the model.
    #     Args:
    #         idx: The initial index of the text to generate.
    #         length: The length of the text to generate.
    #     Returns:
    #         A list of integers representing the generated text.
    #     """
    #     pass

    def save(self, path: str):
        """Save the model to a file."""
        path = self.update_model_path(path)
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load the model from a file."""
        self.load_state_dict(torch.load(path))

    @staticmethod
    def update_model_path(path: str):
        """Updates the model path by appending a timestamp to the path"""
        # First remove file extension
        # If path exists, append timestamp

        if os.path.exists(path):
            path, ext = path.split('.')[0], path.split('.')[1]
            # Then append timestamp
            path = path + '_' + str(datetime.now().strftime("%y%m%d-%H%M"))
            # Finally add file extension
            path = path + '.' + ext

        return path

    def print_param_count(self):
        """Count the number of parameters in the model."""
        print(f"{self.__class__.__name__} has {sum(p.numel() for p in self.parameters()):,} parameters.")
