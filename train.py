import torch
import numpy as np
from data.data_handler import read_in_data, make_char_dict, save_data_as_txt, create_simple_encoder_decoder

if __name__ == '__main__':
    # First we read in the data 'data/asimov/asimov_data_1000.txt'
    char_dict, data = read_in_data('data/asimov/asimov_data_1000.txt')

    # Create the encoder and decoder dictionaries
    encoder_dict, decoder_dict, encode, decode = create_simple_encoder_decoder(char_dict)

    # Load the data into torch tensor
    data = torch.tensor(encode(data), dtype=torch.long)

    print(data.shape)
    print(data[:10])




