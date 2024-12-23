import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from gpt.models.eo_transformer import EncodeOnlyTransformer
from utils.basic_tokeniser import BasicTokeniser
from utils.bpe import BPE
from utils.data_utils import read_in_data
from utils.file_utils import load_config
from utils.train_utils import set_seed

# # Save the training hyperparameters as a  txt file
# save_config(training_hyperparams, 'gpt_config.txt')

# Load the training hyperparameters from the txt file
training_hyperparams = load_config("../gravity_config.txt")

# Set the random seed for reproducibility
# torch.manual_seed(6345789)
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
data_folder = "data/gravity/"
file_path = "examples_same_steps.txt"

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
train_data = pd.read_csv(data_folder + "train_same.csv")
val_data = pd.read_csv(data_folder + "val_same.csv")
test_data = pd.read_csv(data_folder + "test_diff.csv")

# Find the longest question in the training data. This will be used to set the max sequence length
max_seq_len = max(
    max(train_data["question"].apply(lambda x: len(x))),
    max(val_data["question"].apply(lambda x: len(x))),
    max(test_data["question"].apply(lambda x: len(x))),
)

max_ans_len = max(
    max(train_data["answer"].apply(lambda x: len(str(x)))),
    max(val_data["answer"].apply(lambda x: len(str(x)))),
    max(test_data["answer"].apply(lambda x: len(str(x)))),
)

# Convert the data to tensors
# Encode each question and answer.
train_x = []
train_y = []
for i in range(len(train_data)):
    train_x.append(encode(train_data["question"][i]))
    train_y.append(train_data["answer"][i])
    # pad the question with the pad token if they are shorter than the max_seq_len
    if len(train_x[-1]) < max_seq_len:
        train_x[-1] = train_x[-1] + [encoder_dict["<pad>"]] * (
            max_seq_len - len(train_x[-1])
        )


val_x = []
val_y = []
for i in range(len(val_data)):
    val_x.append(encode(val_data["question"][i]))
    val_y.append(val_data["answer"][i])
    # pad the question with the pad token if they are shorter than the max_seq_len
    if len(val_x[-1]) < max_seq_len:
        val_x[-1] = val_x[-1] + [encoder_dict["<pad>"]] * (max_seq_len - len(val_x[-1]))

test_x = []
test_y = []
for i in range(len(test_data)):
    test_x.append(encode(test_data["question"][i]))
    test_y.append(test_data["answer"][i])
    # pad the question with the pad token if they are shorter than the max_seq_len
    if len(test_x[-1]) < max_seq_len:
        test_x[-1] = test_x[-1] + [encoder_dict["<pad>"]] * (
            max_seq_len - len(test_x[-1])
        )


train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y).float()
val_x = torch.tensor(val_x)
val_y = torch.tensor(val_y).float()
test_x = torch.tensor(test_x)
test_y = torch.tensor(test_y).float()

# Create the data loaders
train_data = torch.utils.data.TensorDataset(train_x, train_y)
val_data = torch.utils.data.TensorDataset(val_x, val_y)
test_data = torch.utils.data.TensorDataset(test_x, test_y)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True
)

# update block size to be the max sequence length
block_size = max_seq_len


# Create the model, loss function and optimiser

# Use regression loss function
loss_fn = nn.MSELoss()

model = EncodeOnlyTransformer(
    src_pad=encoder_dict["<pad>"],
    src_sos=encoder_dict["<sos>"],
    vocab_size_enc=len(encoder_dict),
    output_size=1,
    pooling="mean",
    max_seq_len=block_size,
    num_heads=training_hyperparams["num_heads"],
    num_layers=training_hyperparams["num_layers"],
    d_model=training_hyperparams["d_model"],
    d_ff=training_hyperparams["d_ff"],
    dropout_prob=training_hyperparams["dropout_prob"],
    device=device,
)

optimiser = torch.optim.Adam(model.parameters(), lr=lr)


# Define the training loop
def train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    # Only want to loop through a subset of the data_loader as it is too large

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze(1)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


# Set the device
device = torch.device(training_hyperparams["device"])

# Move the model and loss function to the device
model = model.to(device)
loss_fn = loss_fn.to(device)

# Define the optimizer
optimizer = optimiser

train_losses = []
val_losses = []
best_val_loss = float("inf")
counter = 0
# Training loop
for epoch in range(max_iters):
    train_loss = train(model, train_loader, loss_fn, optimizer, device)
    # print(f'Epoch [{epoch+1}/{max_iters}] - Train Loss: {train_loss:.4f}')
    if (epoch + 1) % eval_iters == 0:
        # Perform evaluation on the validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0

            for batch_idx, (inputs, targets) in enumerate(val_loader):
                # inputs, targets = next(iter(val_loader))

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs).squeeze(1)
                loss = loss_fn(outputs, targets)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        train_losses.append(train_loss)

        # Print training and validation loss
        print(
            f"Epoch [{epoch+1}/{max_iters}] - Train Loss: {train_loss:.4f}, Est Val Loss: {val_loss:.4f}"
        )

        # Save the model if the validation loss is the best we've seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, "../gravity_gpt.pt")
            counter = 0

        else:
            counter += 1
            if counter > 2:
                print(f"Stopping early at epoch {epoch+1}")
                break


# Plot the losses
plt.figure(figsize=(12, 8))
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Losses")
plt.savefig("gravity_gpt_losses_same.png")
plt.show()

# Evaluate the model on the test set and plot the results and print the metrics
model.eval()
with torch.no_grad():
    test_loss = 0
    predictions = []
    targets = []
    for batch_idx, (inputs, target) in enumerate(test_loader):
        inputs = inputs.to(device)
        target = target.to(device)

        output = model(inputs).squeeze(1)
        loss = loss_fn(output, target)

        test_loss += loss.item()
        predictions.extend(output.tolist())
        targets.extend(target.tolist())

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    # Plot the predictions vs targets
    plt.figure(figsize=(12, 8))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot(targets, targets, c="r")
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs Predictions")
    plt.savefig("targets_vs_predictions_same.png")
    plt.show()
