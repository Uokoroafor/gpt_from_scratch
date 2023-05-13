# This is a full example of an encoder-decoder transformer model

import os
import time
from typing import Tuple, List, Optional
import torch
from torch import nn
from utils.data_utils import read_in_data, tensor_to_string
from utils.basic_tokeniser import create_simple_encoder_decoder
from utils.dummy_file_generators import save_data_as_txt
from my_models.bigram import BigramModel, BigramModelWithAttention, BigramModelWithAandPE, \
    BigramModelWithAandPEandLN, BigramModelWithAandPEandLNandFFN, BigramModelWithAandPEandLNandFFNandDO, \
    BigramWithTransformerBlocks
import matplotlib.pyplot as plt
from attention_block import TransformerBlock, DecoderTransformerBlock

