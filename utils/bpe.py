# File containing a Byte Pair Encoding Class
from typing import List, Tuple, Optional
from collections import defaultdict, Counter
import re
import pickle as pkl


class BPE:
    # Algorithm is as follows:
    # 1. Create the initial vocabulary
    # 2. Get the counts of pairs of characters
    # 3. Get the most frequent pair of characters
    # 4. Merge the most frequent pair of characters in the vocabulary and create a new entry in the vocabulary
    # 5. Repeat steps 2-4 until the desired vocabulary size is reached
    # Default symbols are <sos> for start of sentence and <eos> for end of sentence, <pad> for padding

    def __init__(
        self,
        data: str,
        vocab_size: Optional[int] = 100000,
        sos: Optional[str] = "<sos>",
        eos: Optional[str] = "<eos>",
        pad: Optional[str] = "<pad>",
    ):
        """Byte Pair Encoding Class
        Args:
            data(str): string to be encoded
            vocab_size(int): Maximum size of the vocabulary
            sos(str): start of sentence symbol
            eos(str): end of sentence symbol
            pad(str): padding symbol
        """
        self.vocab_size = vocab_size
        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.space = " "
        self.lookup_table = None
        self.data = self.get_words(data)
        self.tokens = list(set(data))
        self.vocab = self.create_initial_vocab()

        assert vocab_size > len(
            self.vocab
        ), "Vocab size must be greater than or equal to the size of the initial vocab"
        assert vocab_size >= len(
            set(data)
        ), "Vocab size must be greater than or equal to the number of unique characters in the data"

    @staticmethod
    def get_words(data: str) -> List[List[str]]:
        """Get the words from the data
        Args:
            data(str): string to be encoded
        Returns:
            words(List[List[str]]): list of words where each word is a list of characters
        """
        # Want to split on whitespace and punctuation
        # words = re.findall(r'\w+', data)
        words = re.findall(r"\w+\S*\s*", data)

        words = [list(word) for word in words]

        return words

    def create_initial_vocab(self):
        """Create the initial vocabulary"""
        vocab = self.tokens
        return vocab

    def get_counts(self) -> Counter:
        """Looks at the pairs of tokens and counts their frequency"""
        counts = Counter()
        for word in self.data:
            for i in range(len(word) - 1):
                counts[(word[i], word[i + 1])] += 1
        return counts

    @staticmethod
    def get_most_frequent_pair(counts: Counter) -> Tuple[str, str]:
        """Get the most frequent pair of characters
        Args:
            counts(Counter): Counter object containing the counts of pairs of characters
        Returns:
            most_frequent_pair(Tuple[str, str]): most frequent pair of characters
        """
        most_frequent_pair = counts.most_common(1)[0][0]
        return most_frequent_pair

    def merge_most_frequent_pair(self, most_frequent_pair: Tuple[str, str]) -> None:
        """Merge the most frequent pair of characters in the vocabulary and create a new entry in the vocabulary
        Args:
            most_frequent_pair(Tuple[str, str]): most frequent pair of characters
        """
        self.vocab.append("".join(most_frequent_pair))
        # Also update the data
        # if char[0], char[1] in data, replace with char[0] + char[1]
        for word in self.data:
            # Want to check the sublists without using a for loop since we are modifying the list
            i = 0
            while i < len(word) - 1:
                if (
                    word[i] == most_frequent_pair[0]
                    and word[i + 1] == most_frequent_pair[1]
                ):
                    word[i] = "".join(most_frequent_pair)
                    del word[i + 1]
                else:
                    i += 1

    def train(
        self, num_iters: Optional[int] = 1000, verbose: Optional[bool] = False
    ) -> None:
        """Train the BPE model
        Args:
            num_iters(int): number of iterations to train the model
            verbose(bool): whether to print out the number of tokens in the vocab
        """
        for i in range(num_iters):
            counts = self.get_counts()
            most_frequent_pair = self.get_most_frequent_pair(counts)
            self.merge_most_frequent_pair(most_frequent_pair)

        # Once the model is trained, create a lookup table
        self.lookup_table = self.create_lookup_table()

        # Data file is likely to be large, so once the model is trained, we want to delete the data from the object
        self.data = None

        if verbose:
            print("Training complete")
            self.report_size()

    @staticmethod
    def fill_gaps(my_dict: defaultdict):
        """Makes sure the lookup table goes from 0 to vocab_size"""
        keys = list(my_dict.keys())

        for i, key in enumerate(keys):
            my_dict[key] = i

        return my_dict

    def create_lookup_table(self) -> defaultdict:
        """Create a lookup table for the vocabulary"""
        # pad, sos, eos are the first three entries in the vocab
        lookup_table = defaultdict(int)
        lookup_table[self.pad] = 0
        lookup_table[self.sos] = 1
        lookup_table[self.eos] = 2
        lookup_table[self.space] = 3

        # Space is double counted in the vocab, so we want to remove it
        self.vocab.remove(self.space)

        for i, token in enumerate(self.vocab):
            lookup_table[token] = i + 4

        lookup_table = self.fill_gaps(lookup_table)

        return lookup_table

    def encode(self, data: str) -> List[int]:
        """Encode the data
        Args:
            data(str): string to be encoded
        Returns:
            encoded_data(List[int]): list of integers representing the encoded data
        """
        encoded_data = []
        # Want to encode each word separately and encode using the longest possible token
        # For example, if the word is 'hello', and the vocab is
        # ['lowly'], then we want to encode 'lowly' as 'low' + 'ly'
        # and not 'l' + 'o' + 'w' + 'l' + 'y'
        # So we want to start with the longest token in the vocab and then work our way down
        # to the shortest token
        for word in self.get_words(data):
            i = 0
            while i < len(word):
                for j in range(len(word), i, -1):
                    token = "".join(word[i:j])
                    if token in self.vocab:
                        encoded_data.append(self.lookup_table[token])
                        i = j
                        break
                else:
                    # If we don't find a match, then just encode the character
                    encoded_data.append(self.lookup_table[word[i]])
                    i += 1
        return encoded_data

    def decode(self, encoded_data: List[int]) -> str:
        """Decode the encoded data
        Args:
            encoded_data(List[int]): list of integers representing the encoded data
        Returns:
            decoded_data(str): decoded string
        """
        decoded_data = []
        for token in encoded_data:
            # perform a reverse lookup
            for key, value in self.lookup_table.items():
                if value == token:
                    decoded_data.append(key)
                    break
        return decoded_data

    def decode_words(self, encoded_data: List[int]) -> str:
        """Decode the encoded data
        Args:
            encoded_data(List[int]): list of integers representing the encoded data
        Returns:
            decoded_data(str): decoded string
        """
        decoded_data = []
        for token in encoded_data:
            # perform a reverse lookup
            for key, value in self.lookup_table.items():
                if value == token:
                    decoded_data.append(key)
                    break
        return "".join(decoded_data)

    def save(self, path: str) -> None:
        """Save the BPE model
        Args:
            path(str): path to save the model
        """
        with open(path, "wb") as f:
            pkl.dump(self, f)

    def report_size(self) -> None:
        """Report the size of the lookup table"""
        print("Number of tokens in vocab: {}".format(len(self.lookup_table)))


if __name__ == "__main__":
    # Generate lorem ipsum for 10 lines
    data = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a diam lectus. Sed sit amet ipsum mauris. \n"
    data += "Maecenas congue ligula ac quam viverra nec consectetur ante hendrerit. Donec et mollis dolor. \n"
    data += "Praesent et diam eget libero egestas mattis sit amet vitae augue. Nam tincidunt congue enim, \n"
    data += "ut porta lorem lacinia consectetur. Donec ut libero sed arcu vehicula ultricies a non tortor. \n"
    data += "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean ut gravida lorem."

    # Create the BPE object
    bpe = BPE(data)
    # Train the model
    bpe.train(num_iters=10)

    print(bpe.lookup_table)

    # Encode the data
    encoded_data = bpe.encode("lorem")

    # Decode the data
    decoded_data = bpe.decode_words(encoded_data)

    # Print the encoded data
    print("encoded_data", encoded_data)

    # Print the decoded data
    print("decoded_data", decoded_data)

    # Load new data and try encoding and decoding from text file 'data/FrenchEnglish/text_en_lite' and 'data/FrenchEnglish/text_fr_lite'
    new_data_en = open("../data/FrenchEnglish/text_en_lite", "r").read()
    new_data_fr = open("../data/FrenchEnglish/text_fr_lite", "r").read()

    # Encode the data
    # Create a new BPE object
    bpe_new_en = BPE(new_data_en)
    # Train the model
    bpe_new_en.train(num_iters=1000, verbose=True)
    print(bpe_new_en.lookup_table)
    # Encode the data

    encoded_data_new = bpe_new_en.encode(
        "Fruit flies like a banana. That is, they like a banana that is rotten."
    )
    # Decode the data
    decoded_data_new = bpe_new_en.decode(encoded_data_new)
    # Print the encoded data
    print("encoded_data_new", encoded_data_new)
    # Print the decoded data
    print("decoded_data_new", decoded_data_new)

    # Create a new BPE object
    bpe_new_fr = BPE(new_data_fr)

    # Train the model
    bpe_new_fr.train(num_iters=1000, verbose=True)

    print(bpe_new_fr.lookup_table)
    # Encode the data
    encoded_data_new = bpe_new_fr.encode(
        "Les mouches à fruits aiment les bananes. C'est-à-dire qu'elles aiment une banane qui est pourrie."
    )

    # Decode the data
    decoded_data_new = bpe_new_fr.decode_words(encoded_data_new)

    # Print the encoded data
    print("encoded_data_new", encoded_data_new)

    # Print the decoded data
    print("decoded_data_new", decoded_data_new)
    #
    # Save the English and French Encoder and Decoder
    en_path = "../data/FrenchEnglish/bpe_model_en.pkl"
    fr_path = "../data/FrenchEnglish/bpe_model_fr.pkl"
    bpe_new_en.save(en_path)
    bpe_new_fr.save(fr_path)

    # Load the English and French Encoder and Decoder
    bpe_new_en2 = pkl.load(open(en_path, "rb"))
    bpe_new_fr2 = pkl.load(open(fr_path, "rb"))

    # Test that the loaded models work
    encoded_data_new = bpe_new_en2.encode(
        "Fruit flies like a banana. \n That is, they like a banana that is rotten."
    )
    decoded_data_new = bpe_new_en2.decode_words(encoded_data_new)
    print("encoded_data_new", encoded_data_new)
    print("decoded_data_new", decoded_data_new)

    encoded_data_new = bpe_new_fr2.encode(
        "Les mouches à fruits aiment les bananes. \n C'est-à-dire qu'elles aiment "
        "une banane qui est pourrie."
    )
    decoded_data_new = bpe_new_fr2.decode_words(encoded_data_new)
    print("encoded_data_new", encoded_data_new)
    print("decoded_data_new", decoded_data_new)

# TODO: Update the code to handle unknown words
