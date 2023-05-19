# This is a full example of an encoder-decoder transformer model

from typing import Tuple, Optional
import torch
from torch import nn
from attention_block import TransformerBlock, DecoderTransformerBlock
from my_models.AbstractModelClass import AbstractModelClass


# TODO: Complete Implementation of Full Transformer Model
class Transformer(AbstractModelClass):
    def __init__(self, src_vocab_size, trg_vocab_size, **kwargs):
        """

        Args:
            src_vocab_size: Size of the source vocabulary
            trg_vocab_size: Size of the target vocabulary
            **kwargs: Keyword arguments for the transformer blocks

        """
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.Encoder = TransformerBlock(**kwargs)
        self.Decoder = DecoderTransformerBlock(**kwargs)
        self.num_layers = kwargs['num_layers']
        self.linear = nn.Linear(kwargs['embedding_dim'], trg_vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, src: torch.Tensor, trg: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a forward pass of the transformer model. It is used for training the model. It takes in a source
        tensor and a target tensor and returns the output tensor and the attention tensor. Args: src: Source tensor

        trg: Target tensor

        Returns:
            output: Output tensor
        """
        # if trg is None:
        #     raise ValueError("Target tensor must be provided for training.")
        src = self.Encoder(src)
        output = self.Decoder(trg, src)
        output = self.linear(output)
        output = self.softmax(output)
        return output

    def generate(self, idx: torch.Tensor, length: int) -> torch.Tensor:
        """
        This is a forward pass of the transformer model. It is used for generating text. It takes in an initial
        index and a length and returns the generated text. Args: idx: Initial index of the text to generate

        length: Length of the text to generate

        Returns:
            output: Generated text
        """
        pass

# TODO: Complete Temperature and Top-k Sampling








