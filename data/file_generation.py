from random import random
from typing import Union, List, Dict, Optional

# This is a file for generating dummy data for testing purposes.
# I will encode the rules here and the file will generate random text that follows that rule

# Import the necessary packages
allowable_chars = list(
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=,./<>?;' + ':\'[]{}\\|`~"')
print(allowable_chars)

# Get the letter characters
letters = list(set([char.lower() for char in allowable_chars if char.isalpha()]))
print(letters)

# Get the number characters
numbers = [char for char in allowable_chars if char.isdigit()]
print(numbers)

# Get the punctuation characters
punctuation = [char for char in allowable_chars if not char.isalnum()]
print(punctuation)

# Get the spaces
spaces = [char for char in allowable_chars if char.isspace()]
print(spaces)

char_dict = dict(letters=letters, numbers=numbers, punctuation=punctuation, spaces=spaces)
print(char_dict)

# Define the rules for the dummy data
# 1. The data must be a string
# 2. The data must be at most 10 characters long
# 3. The data must contain at least 1 letter
# 4. The data must contain at least 1 number
# 5. punctuation must be followed by a space
# 6. spaces must be followed by a letter

# Define the function to generate the dummy data
def generate_dummy_data(char_dict: Dict[str, List[str]], min_length: int = 1, max_length: int = 10, seed: Optional[int] = 1111) -> str:
    """Generate dummy data that follows the rules defined above.

    Args:
        char_dict (Dict[str, List[str]]): Dictionary of character types and their corresponding characters.
        min_length (int, optional): Minimum length of the dummy data. Defaults to 1.
        max_length (int, optional): Maximum length of the dummy data. Defaults to 10.
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to 1111.

    Returns:
        str: Dummy data that follows the rules defined above.
    """
    # Initialize the dummy data
    dummy_data = ''

    # Get the number of characters in the dummy data
    num_chars = random.randint(min_length, max_length)

    # Get the first character
    first_char_type = random.choice(list(char_dict.keys()))
    first_char = random.choice(char_dict[first_char_type])
    dummy_data += first_char

    # Get the rest of the characters
    for i in range(num_chars - 1):
        # Get the previous character type
        prev_char_type = first_char_type

        # Get the next character type
        next_char_type = random.choice(list(char_dict.keys()))

        # Get the next character
        next_char = random.choice(char_dict[next_char_type])

        # If the previous character was punctuation, the next character must be a space
        if prev_char_type == 'punctuation':
            next_char = ' ' + next_char

        # If the previous character was a space, the next character must be a letter
        if prev_char_type == 'spaces':
            next_char = random.choice(char_dict['letters']) + next_char

        # Add the next character to the dummy data
        dummy_data += next_char

        # Update the first character type
        first_char_type = next_char_type

    return dummy_data

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
    numbers = [char for char in char_list if char.isdigit()]

    # Get the punctuation characters
    punctuation = [char for char in char_list if not char.isalnum()]

    # Get the spaces
    spaces = [char for char in char_list if char.isspace()]

    char_dict = dict(letters=letters, numbers=numbers, punctuation=punctuation, spaces=spaces)
    return char_dict





