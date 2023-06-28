import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from torch import nn
from gpt.models.gpt_transformer import GPT
from utils.basic_tokeniser import create_simple_encoder_decoder
from utils.data_utils import read_in_data
from utils.train_utils import Trainer
from utils.file_utils import load_config


# training_hyperparams = {
#     'batch_size': 32,
#     'epochs': 1000,
#     'learning_rate': 5e-5,
#     'max_seq_len': 64,
#     'num_heads': 8,
#     'num_layers': 4,
#     'd_model': 128,
#     'd_ff': 128 * 4,
#     'dropout_prob': 0.2,
#     'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#     'eval_every': 100,
#     'eval_iters': 10,
#     'save_every': 100,
# }

# # Save the training hyperparameters as a  txt file
# save_config(training_hyperparams, 'gpt_config.txt')
training_hyperparams = load_config('gpt_config.txt')
# print(training_hyperparams)

torch.manual_seed(6345789)  # Set the random seed for reproducibility
# Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

print('Using device: ', training_hyperparams['device'])
device = training_hyperparams['device']
block_size = training_hyperparams['max_seq_len']
batch_size = training_hyperparams['batch_size']
eval_iters = training_hyperparams['eval_every']
max_iters = training_hyperparams['epochs']
lr = training_hyperparams['learning_rate']

data_folder = 'data/madlibs/'
# data_folder = 'data/gatsby/'

# First we read in the data 'data/asimov/asimov_data_1000.txt'
char_dict, data = read_in_data(data_folder + 'dummy_data_1000_lines.txt')
# char_dict, data = read_in_data(data_folder + 'great_gatsby.txt')

# Create the encoder and decoder dictionaries and the encode and decode functions
encoder_dict, decoder_dict, encode, decode = create_simple_encoder_decoder(char_dict)

# Read in the data
with open(data_folder + 'decoded_train_data.txt', 'r') as f:
    train_data = f.read()
with open(data_folder + 'decoded_val_data.txt', 'r') as f:
    val_data = f.read()
with open(data_folder + 'decoded_test_data.txt', 'r') as f:
    test_data = f.read()


def text_to_tensor2(text: str) -> torch.Tensor:
    """Convert a string of text into a tensor of token indices using sent_tokeniser.
    Args:
        text: A string of text.
    Returns:
        A torch tensor of token indices.
    """
    # Assuming your text data is in the variable 'text'
    sentences = sent_tokenize(text)  # split the text into sentences

    encoded_sentences = [[encoder_dict['<sos>']]]
    for sentence in sentences:
        encoded_sentence = encode(sentence)  # start with <sos>
        # encoded_sentence.append(encoder_dict['\n'])  # pad with <pad> tokens
        encoded_sentences.append(encoded_sentence)

    encoded_sentences.append([encoder_dict['<eos>']])

    # concatenate all encoded sentences
    encoded_text = [token for sentence in encoded_sentences for token in sentence]

    return torch.tensor(encoded_text, dtype=torch.long)


def text_to_tensor(text: str) -> torch.Tensor:
    """Convert a string of text into a tensor of token indices using sent_tokeniser.
    Args:
        text: A string of text.
    Returns:
        A torch tensor of token indices.
    """
    # Assuming your text data is in the variable 'text'
    sentences = sent_tokenize(text)  # split the text into sentences

    encoded_sentences = []
    for sentence in sentences:
        encoded_sentence = [encoder_dict['<sos>']]
        encoded_sentence.extend(encode(sentence))  # start with <sos>
        # encoded_sentence = encode(sentence)  # start with <sos>
        encoded_sentence.extend([encoder_dict['\n']])
        encoded_sentence.extend([encoder_dict['<eos>']])

        encoded_sentences.append(encoded_sentence)

    # encoded_sentences.append([encoder_dict['<eos>']])

    # concatenate all encoded sentences
    encoded_text = [token for sentence in encoded_sentences for token in sentence]

    return torch.tensor(encoded_text, dtype=torch.long)


def text_to_tensor_(text: str) -> torch.Tensor:
    """Convert a string of text into a tensor of token indices using word_tokeniser.
    Args:
        text: A string of text.
        Returns:
            A torch tensor of token indices.
    """
    # First replace spaces with <space> tokens
    text_processed = text.replace(' ', ' _space_ ')
    # Then replace newlines with <newline> tokens
    text_processed = text_processed.replace('\n', ' _newline_ ')
    # Then split into words
    words = word_tokenize(text_processed)
    # Then replace the <space> and <newline> tokens with spaces and newlines
    words = [word.replace('_space_', ' ').replace('_newline_', '\n') for word in words]
    # Then encode the words. Start with <sos> and end with <eos>
    encoded_text = [encoder_dict['<sos>']]
    for word in words:
        encoded_word = encode(word)
        encoded_text.extend(encoded_word)
    encoded_text.append(encoder_dict['<eos>'])

    return torch.tensor(encoded_text, dtype=torch.long)


# train_data_ = text_to_tensor_(train_data)

train_data = text_to_tensor(train_data)
val_data = text_to_tensor(val_data)
test_data = text_to_tensor(test_data)

loss_fn = nn.CrossEntropyLoss(ignore_index=encoder_dict['<pad>'])

model = GPT(
    trg_pad=encoder_dict['<pad>'],
    trg_sos=encoder_dict['<sos>'],
    vocab_size_dec=len(encoder_dict),
    max_seq_len=block_size,
    num_heads=training_hyperparams['num_heads'],
    num_layers=training_hyperparams['num_layers'],
    d_model=training_hyperparams['d_model'],
    d_ff=training_hyperparams['d_ff'],
    dropout_prob=training_hyperparams['dropout_prob'],
    device=device)

optimiser = torch.optim.Adam(model.parameters(), lr=lr)

# Use a learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', factor=0.5, patience=1, verbose=True)

# Create a trainer object
trainer = Trainer(model=model, loss_fn=loss_fn, optimiser=optimiser, training_hyperparameters=training_hyperparams)

# Train the model
model, _, _ = trainer.train(train_data, val_data, save_model=True, plotting=True, verbose=True)

chars = decode(
    model.generate(start_token=encoder_dict['<sos>'] * torch.ones((1, 1), dtype=torch.long), max_length=50, k=6,
                   temp=1.6)[0].tolist())

# Join the characters together and then print the string
print(f"Generating Characters: {''.join(chars)}")
print("-----------------------------------------------------")
