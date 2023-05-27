import os
import random
from typing import List, Dict, Optional
from utils.data_utils import save_data_as_txt


def generate_dummy_char_data(char_dict: Dict[str, List[str]], min_length: Optional[int] = 1,
                             max_length: Optional[int] = 100,
                             num_lines: int = 5, seed: Optional[int] = 1111) -> str:
    """Generate dummy data that follows the rules defined above.

    Args:
        char_dict (Dict[str, List[str]]): Dictionary of character types and their corresponding characters.
        min_length (int, optional): Minimum length of the dummy data. Defaults to 1.
        max_length (int, optional): Maximum length of the dummy data. Defaults to 10.
        num_lines (int, optional): Number of lines of dummy data to generate. Defaults to 5.
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to 1111.

    Returns:
        str: Dummy data that follows the rules defined above."""

    # Set the random seed manually for reproducibility.
    random.seed(seed)

    # Create a unique list of all characters in the dictionary
    char_list = []

    for char_type in char_dict.keys():
        char_list += char_dict[char_type]

    char_list = list(set(char_list))

    # Generate the dummy data
    dummy_data = ''

    for i in range(num_lines):
        # Generate the length of the dummy data
        length = random.randint(min_length, max_length)
        # Generate the dummy data
        for char in range(length):
            # Get the character type but first char must be a letter
            if char == 0:
                char_value = random.choice(char_dict['letters'])
            else:
                char_value = random.choice(char_list)
            # Add the character to the dummy data

            dummy_data += char_value
        # Add a new line to the dummy data
        dummy_data += '\n'
    return dummy_data


def generate_dummy_word_data(pos_dict: Dict[str, List[str]], min_length: Optional[int] = 1,
                             max_length: Optional[int] = 10,
                             num_lines: int = 5, seed: Optional[int] = 1111) -> str:
    """Generate dummy data that follows the rules defined below"""

    # Set the random seed manually for reproducibility.
    random.seed(seed)

    noun_list = pos_dict['nouns']
    verb_list = pos_dict['verbs']
    adj_list = pos_dict['adjectives']
    adv_list = pos_dict['adverbs']
    prep_list = pos_dict['prepositions']
    det_list = pos_dict['determiners']
    conj_list = pos_dict['conjunctions']
    pron_list = pos_dict['pronouns']
    inter_list = pos_dict['interjections']

    # Generate the dummy data
    dummy_data = ''

    for i in range(num_lines):
        # Sentence structure is: det, adj, noun, verb, det, adj, noun.
        dummy_data += random.choice(det_list) + ' '
        dummy_data += random.choice(adj_list) + ' '
        dummy_data += random.choice(noun_list) + ' '
        dummy_data += random.choice(verb_list) + ' '
        dummy_data += random.choice(prep_list) + ' '
        dummy_data += random.choice(det_list) + ' '
        dummy_data += random.choice(adj_list) + ' '
        dummy_data += random.choice(noun_list) + '.\n'

    # Remove the last line break
    dummy_data = dummy_data[:-1]

    return dummy_data


def generate_building_blocks_data(min_length: Optional[int] = 5, max_length: Optional[int] = 100, num_lines: int = 5,
                                  seed: Optional[int] = 1111) -> str:
    """Generate dummy data that follows the rules defined below"""

    # data= 'abcde\n '

    # Rule 1 whenever c is chosen, the next character must be d or b
    # Rule 2 whenever e is chosen, the next character must be a new line
    # Rule 3 whenever a is chosen, the next character cannot be a new line
    # Rule 4 first character in a line cannot be new line or space

    # Set the random seed manually for reproducibility.
    random.seed(seed)

    output_data = ''

    for i in range(num_lines):
        # Generate the length of the dummy data
        length = random.randint(min_length, max_length)
        # Generate the dummy data
        for char in range(length):
            if char == 0:
                char_value = random.choice('abcde')
                output_data += char_value
            elif output_data[-1] == 'c':
                char_value = random.choice('db')
                output_data += char_value
            elif output_data[-1] == 'e':
                char_value = '\n'
                output_data += char_value
            elif output_data[-1] == 'a':
                char_value = random.choice('abcde ')
                output_data += char_value
            else:
                char_value = random.choice('abcde \n')
                output_data += char_value

            if char_value == '\n':
                break
        # Add a new line if the last character is not a new line
        if output_data[-1] != '\n':
            output_data += '\n'

    # Remove the last line break
    output_data = output_data[:-1]
    return output_data


def remove_non_unique_lines(string: str) -> str:
    """Remove non-unique lines from a string.

    Args:
        string (str): The string to remove non-unique lines from.

    Returns:
        str: The string with non-unique lines removed.
    """
    # Split the string into lines
    lines = string.split('\n')
    # Remove the non-unique lines
    unique_lines = list(set(lines))
    # Join the unique lines back together
    string = '\n'.join(unique_lines)
    return string


def shuffle_lines(string: str) -> str:
    """Shuffle the lines of a string.

    Args:
        string (str): The string to shuffle the lines of.

    Returns:
        str: The string with the lines shuffled.
    """
    # Split the string into lines
    lines = string.split('\n')
    # Shuffle the lines
    random.shuffle(lines)
    # Join the lines back together
    string = '\n'.join(lines)
    return string


def determinant_check(string: str) -> str:
    """Checks that the determinants are correct."""
    # Split the string into lines
    lines = string.split('\n')
    # Split the lines into words
    words = [line.split(' ') for line in lines]
    # Check that the determinants are correct
    # if determinant is a, then next word must start with a consonant
    # if determinant is an, then next word must start with a vowel
    for i in range(len(words)):
        for j in range(len(words[i])):
            if words[i][j] == 'a':
                if words[i][j + 1][0] in ['a', 'e', 'i', 'o', 'u']:
                    words[i][j] = 'an'
            if words[i][j] == 'an':
                if words[i][j + 1][0] not in ['a', 'e', 'i', 'o', 'u']:
                    words[i][j] = 'a'
    # Join the words back together
    lines = [' '.join(line) for line in words]
    string = '\n'.join(lines)
    return string


if __name__ == '__main__':
    pass
    # # First we read in the data 'data/asimov/asimov_data_1000.txt'
    # char_dict, data = read_in_data('data/asimov/asimov_data_1000.txt')
    #
    # # Generate the dummy data
    # dummy_data = generate_dummy_char_data(char_dict, min_length=5, max_length=100, num_lines=1000, seed=1111)
    #
    # # Save the data
    # save_data_as_txt(dummy_data, path='../data/asimov/asimov_data_1000.txt')
    #
    # # Create the pos_dict
    # nouns = ['cat', 'dog', 'house', 'car', 'tree', 'book', 'computer', 'phone', 'table', 'chair']
    # verbs = ['runs', 'walks', 'jumps', 'talks', 'speaks', 'writes', 'reads', 'thinks', 'sleeps', 'eats']
    # adjectives = ['quick', 'small', 'tall', 'short', 'fat', 'thin', 'smart', 'robotic', 'large', 'dainty', 'brown',
    #               'blue', 'red', 'green', 'yellow', 'orange', 'purple', 'black', 'white', 'gray']
    # adverbs = ['quickly', 'slowly', 'quietly', 'loudly', 'happily', 'sadly', 'angrily', 'calmly', 'easily', 'hardly']
    # prepositions = ['in', 'on', 'at', 'under', 'over', 'above', 'below', 'behind', 'beside','to']
    # determiners = ['the', 'a', 'an', 'this', 'that', 'my', 'your', 'its', 'our', 'their']
    # conjunctions = ['and', 'but', 'or', 'so', 'yet', 'for', 'nor']
    # pronouns = ['I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']
    # interjections = ['wow', 'oh', 'oops', 'ah', 'hmm', 'huh', 'yuck', 'yikes', 'yay', 'uh']
    #
    # pos_dict = {'nouns': nouns, 'verbs': verbs, 'adjectives': adjectives, 'adverbs': adverbs,
    #             'prepositions': prepositions,
    #             'determiners': determiners, 'conjunctions': conjunctions, 'pronouns': pronouns,
    #             'interjections': interjections}
    #
    # # Generate the dummy data
    # dummy_data = generate_dummy_word_data(pos_dict, min_length=5, max_length=100, num_lines=1000, seed=1111)
    #
    # # Remove any non unique lines
    # dummy_data = shuffle_lines(remove_non_unique_lines(dummy_data))
    #
    # # Check that the determinants are correct
    # dummy_data = determinant_check(dummy_data)
    #
    # # Save the data
    # save_data_as_txt(dummy_data, path='../data/madlibs/dummy_data.txt')
    #
    # basic_data = generate_building_blocks_data(100, 1000, 1111)
    #
    # # Shuffle the lines
    # basic_data = shuffle_lines(basic_data)
    #
    # # Remove any non unique lines
    # basic_data = remove_non_unique_lines(basic_data)
    #
    # print(basic_data)
    #
    # # Save the data
    # save_data_as_txt(basic_data, path='../data/basic_data/basic_data_1000.txt')
