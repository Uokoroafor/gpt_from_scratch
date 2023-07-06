# Utility file for handling data and building the character
# and word-based dictionaries as well as the encoder and decoder functions.
import os
from typing import Union, Dict, Tuple, Callable, Optional, List, Any
import requests as requests
import torch
from utils.basic_tokeniser import make_char_dict
from nltk.tokenize import sent_tokenize, word_tokenize


def read_in_data(
    filepath: str, make_dict: Optional[bool] = True
) -> Union[Tuple[Dict[str, List[str]], str], str]:
    """Read in the data from a file and makes the character dictionary.
    Args:
        filepath (str): The path to the file to read in.
        make_dict (Optional[bool], optional): Whether to make the character dictionary. Defaults to True.
    Returns:
        Tuple[Dict[str, str],str]: The character dictionary ,if make_dict is True, and the data.
        str: The data if make_dict is False.
    """

    # First read in text file
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read()

    if not make_dict:
        return data

    # Make the character dictionary

    char_dict = make_char_dict(data)

    return char_dict, data


def tensor_to_string(tensor: torch.Tensor, decode_func: Callable) -> str:
    """Convert a tensor to a string using the decode function."""
    # Convert the tensor to a list
    tensor_list = tensor.tolist()

    # Decode the tensor
    string = decode_func(tensor_list)

    # Join the list into a string
    string = "".join(string)

    return string


def save_data_as_txt(data: str, path: Optional[str] = None) -> None:
    """Save the data as a txt file. Add one to the name if the file already exists.

    Args:
        data (str): The data to save.
        path (str): The path to save the data to.
    """

    if path is None:
        path = "data/dummy_data.txt"
    else:
        path = path

    while os.path.exists(path):
        # Get the numeric characters from the right side of the string
        numerics = get_numerics_from_string(path)
        # Add one to the numerics
        numerics = int(numerics) + 1
        # Add the numerics to the path
        path = (
            path.replace(str(numerics - 1), str(numerics))
            if numerics > 1
            else path[:-4] + "_1.txt"
        )

    # Save the data as a txt file
    with open(path, "w") as f:
        f.write(data)


def url_to_data(url: str, path: Optional[str] = None) -> str:
    """Get the data from the url.

    Args:
        url (str): The url to get the data from.
        path (str): The location to save the data to.

    Returns:
        str: The data from the url.
    """
    # Read in the data
    data = requests.get(url).text

    # Save the data as a txt file
    if path is not None:
        save_data_as_txt(data, path)

    return data


def get_numerics_from_string(string: str) -> int:
    """Get the numeric characters from the right side of the string.

    Args:
        string (str): The string to get the numeric characters from.

    Returns:
        int: The numeric characters from the right side of the string.
    """
    # Get the numeric characters from the right side of the string
    numerics = ""
    string = string[:-4]
    for char in string[::-1]:
        if char.isdigit():
            numerics += char
        else:
            break
    # Reverse the string
    numerics = numerics[::-1]
    if len(numerics) == 0:
        numerics = 0

    return numerics


def text_to_tensor(
    text: str, tokeniser: Any, add_sos_eos: Optional[bool] = True
) -> torch.Tensor:
    """Convert a string of text into a tensor of token indices using sent_tokeniser.
    Args:
        text: A string of text.
        tokeniser: A tokeniser object - it must have a lookup_table attribute and an encode method.
        add_sos_eos: Whether to add <sos> and <eos> tokens to the start and end of the text.

    Returns:
        A torch tensor of token indices.
    """
    # Assuming your text data is in the variable 'text'
    sentences = sent_tokenize(text)  # split the text into sentences
    encode_fn = tokeniser.encode
    encoder_dict = tokeniser.lookup_table
    sos = encoder_dict[tokeniser.sos]  # get the index of the <sos> token
    eos = encoder_dict[tokeniser.eos]  # get the index of the <eos> token

    encoded_sentences = []
    for sentence in sentences:
        # start with <sos> if add_sos_eos is True
        encoded_sentence = [sos] if add_sos_eos else []
        encoded_sentence.extend(encode_fn(sentence))
        if add_sos_eos:
            # End with <eos> and <newline> tokens if add_sos_eos is True
            encoded_sentence.extend([eos])
            encoded_sentence.extend([encoder_dict["\n"]])

        encoded_sentences.append(encoded_sentence)

    # concatenate all encoded sentences
    encoded_text = [token for sentence in encoded_sentences for token in sentence]

    return torch.tensor(encoded_text, dtype=torch.long)
