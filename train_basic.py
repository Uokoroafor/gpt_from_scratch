import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from my_models.bigram import BigramModel
from utils.data_utils import read_in_data, tensor_to_string
from utils.dummy_file_generators import save_data_as_txt
from utils.my_tokeniser import create_simple_encoder_decoder

if __name__ == '__main__':

    torch.manual_seed(6345789)  # Set the random seed for reproducibility
    # Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

    # Set Hyperparameters
    batch_size = 32  # This is the size of the batch of data that will be processed at once
    block_size = 8  # This is the size of the context window
    max_iters = 10000  # How many iterations to train for
    eval_every = max_iters // 10  # How often to evaluate the model
    lr = 0.001
    eval_iters = 1000  # How many iterations to evaluate for
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Using device: ', device)

    data_folder = 'data/basic_data/'

    # First we read in the data 'data/asimov/asimov_data_1000.txt'
    char_dict, data = read_in_data(data_folder + 'basic_data_1000.txt')

    # Create the encoder and decoder dictionaries and the encode and decode functions
    encoder_dict, decoder_dict, encode, decode = create_simple_encoder_decoder(char_dict)

    if not os.path.exists(data_folder + 'train_data.txt'):
        # Load the data into torch tensor
        data = torch.tensor(encode(data), dtype=torch.long)
        # Apply the decode function to the data when converted to a list
        decoded_data = decode(data.tolist())

        train_data = data[:int(len(data) * 0.7)]
        val_data = data[int(len(data) * 0.7):int(len(data) * 0.9)]
        test_data = data[int(len(data) * 0.9):]

        # Create the train data if the saved file doesn't already exist

        # Create an 70, 20, 10 train, validation, test split
        train_data_str = tensor_to_string(train_data, decode)
        val_data_str = tensor_to_string(val_data, decode)
        test_data_str = tensor_to_string(test_data, decode)

        # Save the data as a text file
        save_data_as_txt(train_data_str, data_folder + 'decoded_train_data.txt')
        save_data_as_txt(val_data_str, data_folder + 'decoded_val_data.txt')
        save_data_as_txt(test_data_str, data_folder + 'decoded_test_data.txt')

    else:
        # Read in the data
        with open(data_folder + 'decoded_train_data.txt', 'r') as f:
            train_data = f.read()
        with open(data_folder + 'decoded_val_data.txt', 'r') as f:
            val_data = f.read()
        with open(data_folder + 'decoded_test_data.txt', 'r') as f:
            test_data = f.read()

        # Load the data into torch tensor
        train_data = torch.tensor(encode(train_data), dtype=torch.long)
        val_data = torch.tensor(encode(val_data), dtype=torch.long)
        test_data = torch.tensor(encode(test_data), dtype=torch.long)

    x = train_data[:block_size]
    y = train_data[1:block_size + 1]
    for t in range(block_size):
        context = x[:t + 1]
        target = y[t]
        print(
            f"when input is '{tensor_to_string(context, decode)}' ({context}), the target is:'{decode([target.tolist()])}'({target})")


    def get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data from the train, validation or test set."""

        if split == 'train':
            data = train_data
        elif split == 'val':
            data = val_data
        elif split == 'test':
            data = test_data
        else:
            raise ValueError(f"Unknown split: '{split}'")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)  # Transfer the data to the GPU if we are using it
        return x, y


    @torch.no_grad()
    def estimate_loss():
        """Evaluate the model on the validation set.
        Returns:
            """
        model.eval()  # Put the model in evaluation mode
        out = {}
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                x, y = get_batch(split)
                _, loss = model(idx=x, target=y)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out


    xb, yb = get_batch('train')

    for b in range(batch_size):  # batch dimension
        for t in range(block_size):  # time dimension
            context = xb[b, :t + 1]
            target = yb[b, t]
            # print(f"when input is {context.tolist()} the target: {target}")

    # Initialize the model
    model = BigramModel(len(encoder_dict), len(encoder_dict))
    model.to(device)  # Transfer the model to the GPU if we are using it

    Optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    # Training loop
    for iter in range(max_iters):
        if iter % eval_every == 0:
            losses = estimate_loss()
            # Print Step, train loss and validation loss
            print(f'At Iteration: {iter}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])

        # Get a batch of data
        xb, yb = get_batch('train')

        # Zero the gradients
        Optimiser.zero_grad()

        # Get the embeddings and the loss (Forward pass)
        embeds, loss = model(idx=xb, target=yb)

        # Backpropagate the loss (Backward pass)
        loss.backward()

        # Take a step with the optimiser
        Optimiser.step()

    chars = decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long), length=1000)[0].tolist())
    # Join the characters together and then print the string
    print(''.join(chars))
    print('---------------------------------')

    # Create x axis values tensor
    x = torch.arange(0, max_iters, eval_every)
    # Plot the losses
    plt.plot(x, train_losses, label='train')
    plt.plot(x, val_losses, label='val')
    plt.title(f'Losses for the {type(model).__name__} model over {max_iters} steps')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
