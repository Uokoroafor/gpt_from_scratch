from typing import Union, List, Dict, Tuple


# This is a file for generating dummy data for testing purposes.
# I will encode the rules here and the file will generate random text that follows that rule


# Define the rules for the dummy data
# 1. The data must be a string
# 2. The data must be at most 10 characters long
# 3. The data must contain at least 1 letter
# 4. The data must contain at least 1 number
# 5. punctuation must be followed by a space
# 6. spaces must be followed by a letter

# List all allowable characters

def make_char_dict(char_list: Union[List[str], str]) -> Dict[str, List[str]]:
    """Make a dictionary of character types and their corresponding characters.

    Args:
        char_list (Union[List[str], str]): List of characters or string of characters.

    Returns:
        Dict[str, List[str]]: Dictionary of character types and their corresponding characters.
    """
    # Get the letter characters
    letters = list(set([char.lower() for char in char_list if char.isalpha()]))

    # Get the numerical characters
    numbers = list(set([char for char in char_list if char.isdigit()]))

    # Get the punctuation characters
    punctuation = list(set([char for char in char_list if not char.isalnum()]))

    # Get the spaces
    spaces = list(set([char for char in char_list if char.isspace()]))

    # Include the newline character if not already included
    new_line = ['\n']

    char_dict = dict(letters=letters, numbers=numbers, punctuation=punctuation, spaces=spaces, new_line=new_line)

    print('Corpus has {} unique letter(s), {} unique numbers(s) and {} unique punctuation(s)'.format(len(letters), len(numbers), len(punctuation)))
    print('Corpus has {} unique characters.'.format(len(set(char_list))))
    return char_dict


def read_in_data(filepath: str) -> Tuple[Dict[str, str], str]:
    """Read in the data from a file and makes the character dictionary.
    Args:
        filepath (str): The path to the file to read in.
    Returns:
        Dict[str, str]: The character dictionary and the data.
    """

    # First read in text file
    with open(filepath, 'r') as f:
        data = f.read()

    # Make the character dictionary
    char_dict = make_char_dict(data)

    return char_dict, data

def create_simple_encoder_decoder(char_dict: Dict[str, List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """This will be a character encoder and decoder for a simple character level language model based on the character dictionary."""
    # Create the encoder and decoder dictionaries
    encoder_dict = dict()
    decoder_dict = dict()

    # Add the special tokens
    encoder_dict['<pad>'] = 0
    decoder_dict[0] = '<pad>'
    encoder_dict['<sos>'] = 1
    decoder_dict[1] = '<sos>'
    encoder_dict['<eos>'] = 2
    decoder_dict[2] = '<eos>'

    # Encode based on the position in the dictionary
    # List all the character values in the dict

    char_list = []
    for char_type in char_dict.keys():
        char_list += char_dict[char_type]

    char_list = list(set(char_list))

    # Add the characters to the encoder and decoder dictionaries
    for i, char in enumerate(char_list):
        encoder_dict[char] = i + 3
        decoder_dict[i + 3] = char


    # create encode, decode functions
    encode = lambda x: [encoder_dict[char.lower()] for char in x]
    decode = lambda x: [decoder_dict[char] for char in x]

    return encoder_dict, decoder_dict, encode, decode


# Define the function to generate the dummy data



if __name__ == '__main__':
    # Read in the data
    char_dict, data = read_in_data('asimov/asimov123.txt')

    # Create the encoder and decoder dictionaries
    encoder_dict, decoder_dict, encode, decode = create_simple_encoder_decoder(char_dict)

    # Test the encode and decode functions:
    print('Original data: ', data[:100])
    print('Encoded data: ', encode(data[:100]))
    print('Decoded data: ', decode(encode(data[:100])))




