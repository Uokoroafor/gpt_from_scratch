# Utility file for handling data and building the character and word-based dictionaries as well as the encoder and
# decoder functions.
from typing import Union, List, Dict, Tuple, Callable, Optional


def make_char_dict(char_list: Union[List[str], str], allow_uppers: Optional[bool] = False) -> Dict[str, List[str]]:
    """Make a dictionary of character types and their corresponding characters.

    Args:
        char_list (Union[List[str], str]): List of characters or string of characters.
        allow_uppers (Optional[bool], optional): Whether to allow uppercase letters. Defaults to False.

    Returns:
        Dict[str, List[str]]: Dictionary of character types and their corresponding characters.
    """
    # Get the letter characters
    if allow_uppers:
        letters = list(set([char for char in char_list if char.isalpha()]))
    else:
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

    print('Corpus has {} unique letter(s), {} unique numbers(s) and {} unique punctuation(s)'.format(len(letters),
                                                                                                     len(numbers),
                                                                                                     len(punctuation)))
    print('Corpus has {} unique characters.'.format(len(set(char_list))))
    return char_dict


def create_simple_encoder_decoder(char_dict: Dict[str, List[str]], add_specials: Optional[bool] = False) -> \
        Tuple[Dict[str, int], Dict[int, str], Callable, Callable]:
    """This will be a character encoder and decoder for a simple character level language model based on the
    character dictionary.

    Args:
        char_dict (Dict[str, List[str]]): The character dictionary.
        add_specials (Optional[bool], optional): Whether to add special tokens. Defaults to False.

    Returns: Tuple[Dict[str, int], Dict[int, str], Callable, Callable]: The encoder and decoder dictionaries and the
    encoder and decoder functions.
    """
    # Create the encoder and decoder dictionaries
    encoder_dic = dict()
    decoder_dic = dict()

    # Add the special tokens if specified
    if add_specials:
        encoder_dic['<pad>'] = 0
        decoder_dic[0] = '<pad>'
        encoder_dic['<sos>'] = 1
        decoder_dic[1] = '<sos>'
        encoder_dic['<eos>'] = 2
        decoder_dic[2] = '<eos>'

    # Encode based on the position in the dictionary
    # List all the character values in the dict

    char_list = []
    for char_type in char_dict.keys():
        char_list += char_dict[char_type]

    char_list = sorted(list(set(char_list)))

    k = len(encoder_dic.keys())

    # Add the characters to the encoder and decoder dictionaries
    for i, char in enumerate(char_list):
        encoder_dic[char] = i + k
        decoder_dic[i + k] = char

    # create encode, decode functions
    # encode = lambda x: [encoder_dic[char.lower()] for char in x]
    # decode = lambda x: [decoder_dic[char] for char in x]

    def encode_func(x: str) -> List[int]:
        """Encode a string of characters.

        Args:
            x (str): String of characters.

        Returns:
            List[int]: List of encoded characters.
        """
        return [encoder_dic[char.lower()] for char in x]

    def decode_func(x: List[int]) -> List[str]:
        """Decode a list of encoded characters.

        Args:
            x (List[int]): List of encoded characters.

        Returns:
            List[str]: List of decoded characters.
        """
        return [decoder_dic[char] for char in x]

    return encoder_dic, decoder_dic, encode_func, decode_func


if __name__ == '__main__':
    # Read in the data
    from utils.data_utils import read_in_data

    char_dict, data = read_in_data('../data/asimov/asimov123.txt')

    # Create the encoder and decoder dictionaries
    encoder_dict, decoder_dict, encode, decode = create_simple_encoder_decoder(char_dict)

    # Test the encode and decode functions:
    print('Original data: ', data[:100])
    print('Encoded data: ', encode(data[:100]))
    print('Decoded data: ', decode(encode(data[:100])))
