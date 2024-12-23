import pandas as pd
import torch
from torch import nn

from gpt.models.eo_transformer import EncodeOnlyTransformer
from utils.basic_tokeniser import BasicTokeniser
from utils.bpe import BPE
from utils.data_utils import read_in_data, make_data_loaders
from utils.file_utils import load_config
from utils.train_utils import PhysicalTrainer, set_seed

# Load the training hyperparameters from the txt file
training_hyperparams = load_config("../bounce_config.txt")

# Set the random seed for reproducibility
set_seed(6_345_789)
# Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0


print("Using device: ", training_hyperparams["device"])
device = training_hyperparams["device"]
block_size = training_hyperparams["max_seq_len"]
batch_size = training_hyperparams["batch_size"]
eval_iters = training_hyperparams["eval_every"]
max_iters = training_hyperparams["epochs"]
lr = training_hyperparams["learning_rate"]

# data_folder = "data/madlibs/"
data_folder = "data/shelf_bounce/all_fixed/"
file_path = "bounce_log_fixed.txt"
function_name = "fixed_bounce"

train_data_path = f"train_{function_name}.csv"
val_data_path = f"val_{function_name}.csv"
test_data_path = f"test_{function_name}.csv"
output_type = "text"  # 'num' or 'text'

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
    gpt_tokeniser = BasicTokeniser(data)

# Create the encoder and decoder dictionaries and the encode and decode functions
encoder_dict, decoder_dict, encode, decode = (
    gpt_tokeniser.lookup_table,
    gpt_tokeniser.reverse_lookup_table,
    gpt_tokeniser.encode,
    gpt_tokeniser.decode,
)

encoding_utils = dict(
    enc_dict=encoder_dict, dec_dict=decoder_dict, encode_fn=encode, decode_fn=decode
)

# Read in the data as pandas dataframes
train_data = pd.read_csv(data_folder + train_data_path, dtype=str)
val_data = pd.read_csv(data_folder + val_data_path, dtype=str)
test_data = pd.read_csv(data_folder + test_data_path, dtype=str)

train_loader, val_loader, test_loader, max_seq_len = make_data_loaders(
    tokeniser=gpt_tokeniser,
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    batch_size=batch_size,
    output=output_type,
    shuffle=True,
)

# update block size to be the max sequence length
block_size = max_seq_len

# Create the model, loss function and optimiser
loss_fn = (
    nn.MSELoss()
    if output_type == "num"
    else nn.CrossEntropyLoss(ignore_index=encoder_dict[gpt_tokeniser.pad])
)

model = EncodeOnlyTransformer(
    src_pad=encoder_dict["<pad>"],
    src_sos=encoder_dict["<sos>"],
    vocab_size_enc=len(encoder_dict),
    output_size=1 if output_type == "num" else len(encoder_dict),
    pooling="max" if output_type == "num" else "none",
    max_seq_len=block_size,
    num_heads=training_hyperparams["num_heads"],
    num_layers=training_hyperparams["num_layers"],
    d_model=training_hyperparams["d_model"],
    d_ff=training_hyperparams["d_ff"],
    dropout_prob=training_hyperparams["dropout_prob"],
    device=device,
)

optimiser = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = None

device = torch.device(training_hyperparams["device"])

# Move the model and loss function to the device
model = model.to(device)
loss_fn = loss_fn.to(device)

trainer = PhysicalTrainer(
    model=model,
    optimiser=optimiser,
    loss_fn=loss_fn,
    training_hyperparameters=training_hyperparams,
    encoding_utils=encoding_utils,
    scheduler=scheduler,
)

model, _, _ = trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    save_model=True,
    plotting=True,
    verbose=True,
    early_stopping=True,
    early_stopping_patience=10,
)

trainer.log_numerical_outputs(
    test_loader, decode, "test_log.txt", output_type=output_type
)
