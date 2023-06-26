from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from gpt.models.gpt_decoder import GPTDecoder


class GPT(nn.Module):
    def __init__(self, trg_pad: int, trg_sos: int, vocab_size_dec: int, d_model: int,
                 d_ff: int, max_seq_len: int, num_layers: Optional[int] = 6, num_heads: Optional[int] = 8,
                 dropout_prob: Optional[float] = 0.1, device: Optional[str] = 'cpu'):
        """ Constructor class for the transformer. It consists of both the encoder and the decoder.
        Args:
            trg_pad (int): Target padding index
            trg_sos (int): Target start of sentence token
            vocab_size_dec (int): Size of the vocabulary of the decoder
            d_model (int): Dimension of the model
            d_ff (int): Hidden dimension of the Feed Forward Layer
            max_seq_len (int): Maximum sequence length
            num_layers (int): Number of decoder layers
            num_heads (int): Number of heads in the multi-head attention
            dropout_prob (float): Dropout probability
            device (str): Device - 'cpu' or 'cuda'
        """
        super(GPT, self).__init__()
        self.trg_pad = trg_pad
        self.trg_sos = trg_sos
        self.decoder = GPTDecoder(vocab_size_dec, d_model, max_seq_len, num_layers, num_heads, d_ff, dropout_prob)
        self.device = device

    def forward(self, trg: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer
        Args:
            trg (torch.Tensor): Target tensor of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        trg_mask = self.get_trg_mask(trg)
        decoder_output = self.decoder(trg, trg_mask)
        return decoder_output

    def get_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """Create target mask
        Args:
            trg (torch.Tensor): Target tensor of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Target mask tensor of shape (batch_size, seq_len, seq_len)
        """
        # What to ignore the padding tokens
        trg_pad_mask = (trg != self.trg_pad).unsqueeze(-2)  # (batch_size, 1, seq_len)
        trg_len = trg.shape[1]
        # What to ignore the future tokens (i.e. tokens that are not yet predicted)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # Final mask ignores both padding and future tokens
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def greedy_generate(self, start_token: int, max_length: int) -> torch.Tensor:
        """
        Generate a sequence given a start token
        Args:
            start_token (int): Start token
            max_length (int): Maximum length of the sequence to generate
        Returns:
            torch.Tensor: Generated sequence
        """
        with torch.no_grad():
            generated = torch.tensor([start_token], dtype=torch.long, device=self.device).unsqueeze(0)
            for _ in range(max_length):
                output = self.forward(generated)
                next_token = output.argmax(2)[:, -1].unsqueeze(1)
                generated = torch.cat((generated, next_token), dim=1)
        return generated

    def top_k_generate(self, start_token: int, max_length: int, k: int, temp: Optional[float] = 1.0) -> torch.Tensor:
        """
        Generate a sequence given a start token
        Args:
            start_token (int): Start token
            max_length (int): Maximum length of the sequence to generate
            k (int): Number of top tokens to consider
            temp (float): Temperature to apply to the logits. If provided, it must be greater than 0.0
        Returns:
            torch.Tensor: Generated sequence
        """
        with torch.no_grad():
            generated = torch.tensor([start_token], dtype=torch.long, device=self.device).unsqueeze(0)
            for _ in range(max_length):
                output = self.forward(generated)

                # apply a temperature to the output logits
                assert temp > 0.0, "Temperature must be greater than 0.0"
                output = output[:, -1, :] / temp

                # apply a softmax to transform the logits to probabilities
                probabilities = F.softmax(output[:, -1, :], dim=-1)
                # filter top-k tokens
                top_k_probs, top_k_indices = torch.topk(probabilities, k=k, dim=-1)
                # sample from the top_k tokens
                # next_token = torch.multinomial(top_k_probs, num_samples=1)
                # # add the sampled token to the generated sequence
                # generated = torch.cat((generated, top_k_indices[0, next_token].unsqueeze(0).unsqueeze(0)), dim=1)

                next_token = torch.multinomial(top_k_probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
            return generated
