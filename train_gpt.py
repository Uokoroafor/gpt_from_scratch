import torch
from torch import nn
from gpt.models.gpt_transformer import GPT
from utils.basic_tokeniser import BasicTokeniser
from utils.data_utils import read_in_data, text_to_tensor
from utils.train_utils import Trainer
from utils.file_utils import load_config
from utils.bpe import BPE

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

# Load the training hyperparameters from the txt file
training_hyperparams = load_config("gpt_config.txt")

# Set the random seed for reproducibility
torch.manual_seed(6345789)
# Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

print("Using device: ", training_hyperparams["device"])
device = training_hyperparams["device"]
block_size = training_hyperparams["max_seq_len"]
batch_size = training_hyperparams["batch_size"]
eval_iters = training_hyperparams["eval_every"]
max_iters = training_hyperparams["epochs"]
lr = training_hyperparams["learning_rate"]

# data_folder = "data/madlibs/"
data_folder = 'data/gatsby/'
file_path = 'great_gatsby.txt'

use_bpe = False  # Set to True to use BPE, False to use a character encoder/decoder

# Read in the data
data = read_in_data(data_folder + file_path, make_dict=False)

# Create the tokeniser
if use_bpe:
    bpe = BPE(data)
    # Train for 50 iterations
    bpe.train(50)
    gpt_tokeniser = bpe
else:
    # Use BasicTokeniser for char-level encoding
    basic_tokeniser = BasicTokeniser(data)
    gpt_tokeniser = basic_tokeniser

# Create the encoder and decoder dictionaries and the encode and decode functions
encoder_dict, decoder_dict, encode, decode = gpt_tokeniser.lookup_table, gpt_tokeniser.reverse_lookup_table, \
    gpt_tokeniser.encode, gpt_tokeniser.decode

encoding_utils = dict(enc_dict=encoder_dict, dec_dict=decoder_dict, encode_fn=encode, decode_fn=decode)

# Read in the data
with open(data_folder + "decoded_train_data.txt", "r", encoding="utf-8") as f:
    train_data = f.read()
with open(data_folder + "decoded_val_data.txt", "r", encoding="utf-8") as f:
    val_data = f.read()
with open(data_folder + "decoded_test_data.txt", "r", encoding="utf-8") as f:
    test_data = f.read()

train_data = text_to_tensor(train_data, gpt_tokeniser)
val_data = text_to_tensor(val_data, gpt_tokeniser)
test_data = text_to_tensor(test_data, gpt_tokeniser)

# Create the model, loss function and optimiser
loss_fn = nn.CrossEntropyLoss(ignore_index=encoder_dict[gpt_tokeniser.pad])

model = GPT(
    trg_pad=encoder_dict["<pad>"],
    trg_sos=encoder_dict["<sos>"],
    vocab_size_dec=len(encoder_dict),
    max_seq_len=block_size,
    num_heads=training_hyperparams["num_heads"],
    num_layers=training_hyperparams["num_layers"],
    d_model=training_hyperparams["d_model"],
    d_ff=training_hyperparams["d_ff"],
    dropout_prob=training_hyperparams["dropout_prob"],
    device=device,
)

optimiser = torch.optim.Adam(model.parameters(), lr=lr)

# Create a trainer object
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimiser=optimiser,
    training_hyperparameters=training_hyperparams,
    encoding_utils=encoding_utils,
)

# Train the model
model, _, _ = trainer.train(
    train_data, val_data, save_model=True, plotting=True, verbose=True,early_stopping=True
)

sampled_chars = decode(
    model.generate(
        start_token=model.trg_sos * torch.ones((1, 1), dtype=torch.long),
        max_length=100,
        k=6,
        temp=1.6,
    )[0].tolist()
)

greedy_chars = decode(
    model.generate(
        start_token=model.trg_sos * torch.ones((1, 1), dtype=torch.long),
        max_length=100,
        sampled=False,
    )[0].tolist()
)

# Join the characters together and then print the string
print(sampled_chars)
print(f"Generating Characters with sampling: {''.join(sampled_chars)}")
print("-----------------------------------------------------")
print(greedy_chars)
print(f"Generating Characters without sampling: {''.join(greedy_chars)}")
print("-----------------------------------------------------")
