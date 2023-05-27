# This is a full example of an encoder-decoder transformer model

from typing import Tuple, Optional
import torch
from torch import nn
from my_models.attention_block import TransformerBlock, DecoderTransformerBlock
from my_models.AbstractModelClass import AbstractModelClass


# TODO: Complete Implementation of Full Transformer Model

class InputAndPositionalEncoding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        # First create the embedding layer from vocab_size to embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = self.create_positional_encoding()

        # print('InputAndPositionalEncoding initialised')
        # print('vocab_size: ', vocab_size)
        # print('embedding_dim: ', embedding_dim)
        # print('max_seq_len: ', max_seq_len)

    def create_positional_encoding(self):
        """ Create the positional encoding matrix """

        return nn.Embedding(num_embeddings=self.max_seq_len, embedding_dim=self.embedding_dim)
        # Use the formula from the paper




    def forward(self, x):
        """ Forward pass of the input and positional encoding """
        # Get the shape of the input
        batch_size, seq_len = x.shape

        # print('batch_size: ', batch_size)
        # print('seq_len: ', seq_len)
        #
        # if x.max() >= self.vocab_size:
        #     print('x.max(): ', x.max())
        #     print('self.vocab_size: ', self.vocab_size)
        # assert x.max() < self.vocab_size, 'Input is greater than the vocabulary size'

        # Get the embedding of the input
        x = self.embedding(x)

        # print('new x.shape: ', x.shape)
        # Get the positional encoding
        pos = torch.arange(seq_len)  # .repeat(batch_size, 1)
        # print('pos.shape: ', pos.shape)
        # print('x.shape: ', x.shape)
        pos = self.positional_encoding(pos)
        # Add the positional encoding to the input
        x = x + pos

        # # Input Embedding
        # x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        return x


# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super().__init__()
#
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#
#         pe[:, 0::2] = torch.sin(position * div_term)
#
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         pe = pe.unsqueeze(0).transpose(0, 1)
#
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#
#         x = x + self.pe[:x.size(0), :]
#
#         return self.dropout(x)


class Transformer(AbstractModelClass):
    def __init__(self, src_vocab_size, trg_vocab_size, embedding_dim=256, num_layers=2, num_heads=4, dropout_prob=0.2,
                 max_seq_length=100):
        """

        Args:
            src_vocab_size: Size of the source vocabulary
            trg_vocab_size: Size of the target vocabulary
            embedding_dim: Dimension of the embedding
            num_layers: Number of layers in the encoder and decoder
            num_heads: Number of heads in the multi-head attention
            dropout_prob: Dropout probability
            max_seq_length: Maximum sequence length

        """
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.max_seq_length = max_seq_length
        self.encoder_embedding = InputAndPositionalEncoding(src_vocab_size, embedding_dim, max_seq_length)
        self.decoder_embedding = InputAndPositionalEncoding(trg_vocab_size, embedding_dim, max_seq_length)
        self.encoder_blocks = nn.ModuleList([TransformerBlock(embedding_dim=embedding_dim, output_dim=embedding_dim,
                                                              num_heads=num_heads, dropout_prob=dropout_prob) for _ in
                                             range(num_layers)])
        self.decoder_blocks = nn.ModuleList([DecoderTransformerBlock(embedding_dim=embedding_dim,
                                                                     output_dim=embedding_dim, num_heads=num_heads,
                                                                     dropout_prob=dropout_prob) for _ in
                                             range(num_layers)])
        self.linear = nn.Linear(embedding_dim, trg_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src: torch.Tensor, trg: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of our model on some input and target text. This is used during training.
            Args:
                src: Source tensor
                trg: Target tensor

            Returns:
                The predicted tensor

        """
        # Get the embedding of the input
        src = self.encoder_embedding(src)
        # Pass the embedding through the encoder blocks
        for encoder_block in self.encoder_blocks:
            src = encoder_block(src)

        # If the target is not given, then we are in inference mode
        if trg is None:
            # Create a tensor to hold the predicted words
            trg = torch.zeros((src.shape[0], self.max_seq_length)).long()
            # Populate the first word with the start token
            trg[:, 0] = 1
            # Get the embedding of the target
            trg = self.decoder_embedding(trg)
            # Pass the embedding through the decoder blocks
            for decoder_block in self.decoder_blocks:
                trg = decoder_block(trg, src)
            # Get the output
            output = self.linear(trg)
            # Get the softmax of the output
            output = self.softmax(output)
            return output

        else:
            # if trg is given, then we are in training mode

            # Get the embedding of the target
            trg = self.decoder_embedding(trg)

            # Pass the embedding through the decoder blocks
            for decoder_block in self.decoder_blocks:
                trg = decoder_block(trg, src)
            # Get the output
            output = self.linear(trg)
            # Get the softmax of the output
            output = self.softmax(output)
            return output

    def translate(self, src: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of our model on some input text. This is used during inference.
        Args:
            src: Source tensor

        Returns:
            The predicted tensor """
        pass
        # TODO: Decide what object should house the method

        # Padding = 0
        # <sos> = 1
        # <eos> = 2

        # # Get the embedding of the input
        # src = self.encoder_embedding(src)
        # # Pass the embedding through the encoder blocks
        # for encoder_block in self.encoder_blocks:
        #     src = encoder_block(src)
        #
        # # Create a tensor to hold the predicted words
        # out = torch.zeros((src.shape[0], self.max_seq_length)).long()
        # # The first input to the decoder is the <sos> token
        # out[:, 0] = 1
        # # Then we loop over the maximum sequence length for each passed sequence. Stop at max_seq_length - 1 or when we predict the <eos> token
        # for idx in range(1, self.max_seq_length):
        #     # Get the embedding of the target
        #     trg = self.decoder_embedding(out)
        #     # Pass the embedding through the decoder blocks
        #     for decoder_block in self.decoder_blocks:
        #         trg = decoder_block(trg, src)
        #     # Get the output
        #     output = self.linear(trg)
        #     # Get the softmax of the output
        #     output = self.softmax(output)
        #     # Get the predicted next word by getting the index of the highest probability
        #     output = output.argmax(2)
        #     out[:, idx] = output[:, idx]
        #     print('predicting idx: ', idx)
        #     if output[:, idx] == 2:
        #         break
        # # Return the predicted words
        # return out

# TODO: Complete Temperature and Top-k Sampling
