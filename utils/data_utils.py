# Utility file for handling data and building the character
# and word-based dictionaries as well as the encoder and decoder functions.
import os
from typing import Union, Dict, Tuple, Callable, Optional, List, Any
import requests as requests
import torch
from nltk.tokenize import sent_tokenize
from utils.basic_tokeniser import make_char_dict
import pandas as pd
import random


def read_in_data(
        filepath: str, make_dict: Optional[bool] = True
) -> Union[Tuple[Dict[str, List[str]], str], str]:
    """Read in the data from a file and makes the character dictionary.
    Args:
        filepath (str): The path to the file to read in.
        make_dict (Optional[bool], optional): Whether to make the character dictionary. Defaults to True.
    Returns:
        Tuple[Dict[str, str],str]: The character dictionary, if make_dict is True, and the data.
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


def text_to_tensor_(
        text: str, tokeniser: Any, add_sos_eos: Optional[bool] = True
) -> torch.Tensor:
    """Convert a string of text into a tensor of token indices using sent_tokeniser. This version adds <sos> and <eos>
    tokens to the start and end of each sentence. It also adds a <newline> token to the end of each sentence.
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


def text_to_tensor(
        text: str, tokeniser: Any, add_sos_eos: Optional[bool] = True) -> torch.Tensor:
    """Convert a string of text into a tensor of token indices using sent_tokeniser.
    Args:
        text: A string of text.
        tokeniser: A tokeniser object - it must have a lookup_table attribute and an encode method.
        add_sos_eos: Whether to add <sos> and <eos> tokens to the start and end of the text.

    Returns:
        A torch tensor of token indices.
    """
    encode_fn = tokeniser.encode
    encoder_dict = tokeniser.lookup_table
    sos = encoder_dict[tokeniser.sos]  # get the index of the <sos> token
    eos = encoder_dict[tokeniser.eos]  # get the index of the <eos> token

    encoded_text = []
    if add_sos_eos:
        encoded_text.append(sos)

    encoded_text.extend(encode_fn(text))  # encode the entire text

    if add_sos_eos:
        encoded_text.append(eos)

    return torch.tensor(encoded_text, dtype=torch.long)


# def data_prep(folder_loc: str, file_name: str, line_delimiter: str, ans_delimiter: str,
#               split: Optional[List[float]] = None, save_indices: Optional[bool] = False,
#               split_method: Optional[str] = "train_val_test") -> None:
#     """ Prepares the data for training and testing.
#
#     Args:
#         folder_loc (str): The location of the folder containing the data.
#         file_name (str): The name of the file containing the data.
#         line_delimiter (str): The text to split the lines on.
#         ans_delimiter (str): The text to split the answers on.
#         split (List[float]): The split to use for the training and testing data.
#         save_indices (bool): Whether to save the indices of the training and testing data.
#         split_method (str): The method to use for splitting the data. Options are "train_val_test", "train_test"
#         and "train_val".
#     """
#     # Read in the data
#     if not folder_loc.endswith("/"):
#         folder_loc += "/"
#
#     data = read_in_data(os.path.join(folder_loc, file_name), make_dict=False)
#
#     # Split into questions and answers
#     data = [line.split(ans_delimiter) for line in data.split(line_delimiter)]
#
#     if save_indices:
#         assert split is not None, "A split must be provided when saving indices."
#
#     if split_method not in ["train_val_test", "train_val"]:
#         raise ValueError("split_method must be one of 'train_val_test' or 'train_val'.")
#
#     if save_indices:
#         # Randomly shuffle the indices
#         indices = list(range(len(data)))
#         random.shuffle(indices)
#
#         assert sum(split) == 1, "The split must sum to 1."
#
#         tr_weights, val_weights, test_weights = split
#
#         # Split the indices into train, validation and test sets
#         train_indices = indices[:int(tr_weights * len(indices))]
#         val_indices = indices[int(tr_weights * len(indices)):int((tr_weights + val_weights) * len(indices))]
#
#         if split_method == "train_val_test":
#             test_indices = indices[int((tr_weights + val_weights) * len(indices)):]
#
#
#         # Save the indices
#         pd.DataFrame(train_indices).to_csv(os.path.join(folder_loc, "train_indices.csv"), index=False)
#         pd.DataFrame(val_indices).to_csv(os.path.join(folder_loc, "val_indices.csv"), index=False)
#
#         if split_method == "train_val_test":
#             pd.DataFrame(test_indices).to_csv(os.path.join(folder_loc, "test_indices.csv"), index=False)
#     else:
#         # If index files already exist, use them
#
#         train_indices = pd.read_csv(os.path.join(folder_loc, "train_indices.csv"))
#         val_indices = pd.read_csv(os.path.join(folder_loc, "val_indices.csv"))
#
#         if split_method == "train_val_test":
#             test_indices = pd.read_csv(os.path.join(folder_loc, "test_indices.csv"))
#
#
#     # Split the data into training, validation, and testing data
#     train_data = [data[i] for i in train_indices]
#     val_data = [data[i] for i in val_indices]
#
#     if split_method == "train_val_test":
#         test_data = [data[i] for i in test_indices]
#
#     # Save the data as csv files
#     pd.DataFrame(train_data).to_csv(os.path.join(folder_loc, "train_data.csv"), index=False)
#     pd.DataFrame(val_data).to_csv(os.path.join(folder_loc, "val_data.csv"), index=False)
#
#     if split_method == "train_val_test":
#         pd.DataFrame(test_data).to_csv(os.path.join(folder_loc, "test_data.csv"), index=False)
def data_prep(folder_loc: str, file_name: str, line_delimiter: str, ans_delimiter: str,
              split: Optional[List[float]] = None, save_indices: Optional[bool] = False,
              split_method: Optional[str] = "train_val_test") -> None:
    """ Prepares the data for training and testing.

    Args:
        folder_loc (str): The location of the folder containing the data.
        file_name (str): The name of the file containing the data.
        line_delimiter (str): The text to split the lines on.
        ans_delimiter (str): The text to split the answers on.
        split (List[float]): The split to use for the training and testing data.
        save_indices (bool): Whether to save the indices of the training and testing data.
        split_method (str): The method to use for splitting the data. Options are "train_val_test" and "train_val".
    """
    # Read in the data
    if not folder_loc.endswith("/"):
        folder_loc += "/"

    data = read_in_data(os.path.join(folder_loc, file_name), make_dict=False)

    # Split into questions and answers
    data = [line.split(ans_delimiter) for line in data.split(line_delimiter)]

    train_indices, val_indices, test_indices = None, None, None
    train_data, val_data, test_data = None, None, None

    if save_indices:
        if split is None:
            raise ValueError("A split must be provided when saving indices.")

        assert split_method == "train_val_test", "Indices can only be saved for 'train_val_test' method."

        # Randomly shuffle the indices
        indices = list(range(len(data)))
        random.shuffle(indices)

        assert sum(split) == 1, "The split must sum to 1."

        tr_weights, val_weights, test_weights = split

        # Split the indices into train, validation and test sets
        train_indices = indices[:int(tr_weights * len(indices))]
        val_indices = indices[int(tr_weights * len(indices)):int((tr_weights + val_weights) * len(indices))]
        test_indices = indices[int((tr_weights + val_weights) * len(indices)):]

        # Save the indices
        pd.DataFrame(train_indices).to_csv(os.path.join(folder_loc, "train_indices.csv"), index=False)
        pd.DataFrame(val_indices).to_csv(os.path.join(folder_loc, "val_indices.csv"), index=False)
        pd.DataFrame(test_indices).to_csv(os.path.join(folder_loc, "test_indices.csv"), index=False)
    else:
        # If index files already exist, use them

        train_indices = pd.read_csv(os.path.join(folder_loc, "train_indices.csv"))
        val_indices = pd.read_csv(os.path.join(folder_loc, "val_indices.csv"))

        if split_method == "train_val_test":
            test_indices = pd.read_csv(os.path.join(folder_loc, "test_indices.csv"))

    # Split the data into training, validation, and testing data
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]

    if split_method == "train_val_test":
        test_data = [data[i] for i in test_indices]

    # Save the data as csv files
    pd.DataFrame(train_data, columns=["question", "answer"]).to_csv(os.path.join(folder_loc, "train_data.csv"),
                                                                    index=False)
    pd.DataFrame(val_data, columns=["question", "answer"]).to_csv(os.path.join(folder_loc, "val_data.csv"),
                                                                  index=False)

    if split_method == "train_val_test":
        pd.DataFrame(test_data, columns=["question", "answer"]).to_csv(os.path.join(folder_loc, "test_data.csv"),
                                                                       index=False)
