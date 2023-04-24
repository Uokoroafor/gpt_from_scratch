import os

import torch
import numpy as np
from data.data_handler import read_in_data, make_char_dict, create_simple_encoder_decoder, tensor_to_string
from data.dummy_file_generators import save_data_as_txt
from my_models.bigram import BigramModel

if __name__ == '__main__':

    # First we read in the data 'data/asimov/asimov_data_1000.txt'
    char_dict, data = read_in_data('data/madlibs/dummy_data_1000_lines.txt')

    # Create the encoder and decoder dictionaries and the encode and decode functions
    encoder_dict, decoder_dict, encode, decode = create_simple_encoder_decoder(char_dict)

    if not os.path.exists('data/madlibs/train_data.txt'):
        # Load the data into torch tensor
        data = torch.tensor(encode(data), dtype=torch.long)
        # Apply the decode function to the data when converted to a list
        decoded_data = decode(data.tolist())



        train_data = data[:int(len(data) * 0.7)]
        val_data = data[int(len(data) * 0.7):int(len(data) * 0.9)]
        test_data = data[int(len(data) * 0.9):]

        # Create the train data if the saved file doesn't already exist

        # Create an 70, 20, 10 train, validation, test split
        train_data_str = tensor_to_string(train_data,decode)
        val_data_str = tensor_to_string(val_data,decode)
        test_data_str = tensor_to_string(test_data,decode)

        # Save the data as a text file
        save_data_as_txt(train_data_str, 'data/madlibs/decoded_train_data.txt')
        save_data_as_txt(val_data_str, 'data/madlibs/decoded_val_data.txt')
        save_data_as_txt(test_data_str, 'data/madlibs/decoded_test_data.txt')

    else:
        # Read in the data
        with open('data/madlibs/decoded_train_data.txt', 'r') as f:
            train_data = f.read()
        with open('data/madlibs/decoded_val_data.txt', 'r') as f:
            val_data = f.read()
        with open('data/madlibs/decoded_test_data.txt', 'r') as f:
            test_data = f.read()


        # Load the data into torch tensor
        train_data = torch.tensor(encode(train_data), dtype=torch.long)
        val_data = torch.tensor(encode(val_data), dtype=torch.long)
        test_data = torch.tensor(encode(test_data), dtype=torch.long)

    block_size = 8  # Arbitrary block size
    x = train_data[:block_size]
    y = train_data[1:block_size + 1]
    for t in range(block_size):
        context = x[:t + 1]
        target = y[t]
        print(f"when input is '{tensor_to_string(context, decode)}' ({context}), the target is:'{decode([target.tolist()])}'({target})")

        # print(f"when the input is {tensor_to_string(context, decode)} the target is: {decode([target.tolist()])}")


    # Initialize the model
    # model = BigramModel(len(encoder_dict),len(encoder_dict))









