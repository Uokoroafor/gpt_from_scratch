import os
import time
from typing import Tuple, List, Optional

import torch
from torch import nn

from utils.data_handler import read_in_data, tensor_to_string
from utils.my_tokeniser import create_simple_encoder_decoder
from utils.dummy_file_generators import save_data_as_txt
from my_models.bigram import BigramModel, BigramModelWithAttention, BigramModelWithAandPE, \
    BigramModelWithAandPEandLN, BigramModelWithAandPEandLNandFFN, BigramModelWithAandPEandLNandFFNandDO
import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.manual_seed(6345789)  # Set the random seed for reproducibility
    # Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

    # Set Hyperparameters
    batch_size = 32  # This is the size of the batch of data that will be processed at once
    block_size = 64  # This is the size of the context window
    max_iters = 1000  # How many iterations to train for
    eval_every = max_iters // 10  # How often to evaluate the model
    embedding_dim = 256  # The size of the embedding dimension
    lr = 3e-4
    eval_iters = 100  # How many iterations to evaluate for
    dropout_prob = 0.2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Using device: ', device)

    data_folder = 'data/madlibs/'

    # First we read in the data 'data/asimov/asimov_data_1000.txt'
    char_dict, data = read_in_data(data_folder + 'dummy_data_1000_lines.txt')

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
            f"when input is '{tensor_to_string(context, decode)}' ({context}), "
            f"the target is:'{decode([target.tolist()])}'({target})")


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
    def estimate_loss(model_):
        """Evaluate the model on the validation set.
        Returns:
            """
        model_.eval()  # Put the model in evaluation mode
        out = {}
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                x, y = get_batch(split)
                _, loss = model_(idx=x, target=y)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        model_.train()
        return out


    def bigram_training_loop(model: nn.Module, max_iters: int, eval_every: int = 1000, plots: Optional[bool] = True,
                             verbose: Optional[bool] = False) -> Tuple[nn.Module, List[float], List[float]]:
        """Train a model for a number of iterations and evaluate it every `eval_every` iterations.
        Returns:
            model: The trained model.
            train_losses: The training losses for each evaluation step.
            val_losses: The validation losses for each evaluation step.
        """

        model.print_param_count()  # Print the number of parameters in the model
        Optimiser = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        val_losses = []

        # Training loop
        print(f"Training {type(model).__name__} for {max_iters} iterations...")
        # Measure the time taken for the training
        start_time = time.time()
        last_time = start_time
        for i in range(max_iters):
            if i % eval_every == 0:
                losses = estimate_loss(model)
                # Print Step, train loss and validation loss
                if verbose:
                    print(f'At Iteration: {i}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')
                    # Print the time taken for the last eval_every iterations
                    print(f'Time taken for last {eval_every} iterations: {(time.time() - last_time):.2f} seconds')
                    last_time = time.time()
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

        # if verbose:
        #     chars = decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long), length=100)[0].tolist())
        #     # Join the characters together and then print the string
        #     print(''.join(chars))

        if plots:
            # Create x axis values tensor
            x = torch.arange(1, max_iters + 1, eval_every)
            # Plot the losses
            plt.plot(x, train_losses, label='train')
            plt.plot(x, val_losses, label='val')
            plt.title(f'Losses for the {type(model).__name__} model over {max_iters} steps')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        return model, train_losses, val_losses


    models = [
        BigramModel(len(encoder_dict), embedding_dim=len(encoder_dict)),
        BigramModelWithAttention(len(encoder_dict), embedding_dim=embedding_dim, block_size=block_size),
        BigramModelWithAandPE(len(encoder_dict), embedding_dim=embedding_dim, block_size=block_size),
        BigramModelWithAandPEandLN(len(encoder_dict), embedding_dim=embedding_dim, block_size=block_size),
        BigramModelWithAandPEandLNandFFN(len(encoder_dict), embedding_dim=embedding_dim, block_size=block_size),
        BigramModelWithAandPEandLNandFFNandDO(len(encoder_dict), embedding_dim=embedding_dim,
                                              block_size=block_size, dropout_prob=dropout_prob),
    ]

    trained_models = []

    for model in models:
        model, train_losses, val_losses = bigram_training_loop(model, max_iters=max_iters, eval_every=eval_every,
                                                               plots=True, verbose=True)
        print(f"Model: {type(model).__name__}")
        print(f"Train Loss: {train_losses[-1]}")
        print(f"Val Loss: {val_losses[-1]}")
        model.save(f"saved_models/{type(model).__name__}.pt")
        trained_models.append(model)
        print("-----------------------------------------------------")

    # Generate a sample
    for model in trained_models:
        print(f"Generating for Model: {type(model).__name__}")
        chars = decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long), length=100)[0].tolist())
        # Join the characters together and then print the string
        print(''.join(chars))
        print("-----------------------------------------------------")
