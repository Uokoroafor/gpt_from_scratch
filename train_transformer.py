# Test file for training full transformer model on a translation task
# Imports
import random
import time
import torch
from matplotlib import pyplot as plt
from torch import nn
from my_models.my_transformer import Transformer
from utils.basic_tokeniser import create_simple_encoder_decoder
from utils.data_utils import read_in_data

if __name__ == '__main__':
    torch.manual_seed(6345789)  # Set the random seed for reproducibility
    # Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

    # Set Hyperparameters
    batch_size = 64  # This is the size of the batch of data that will be processed at once
    block_size = 64  # This is the size of the context window
    max_iters = 1000  # How many iterations to train for
    eval_every = max_iters // 10  # How often to evaluate the model
    embedding_dim = 256  # The size of the embedding dimension
    lr = 3e-4
    eval_iters = 20  # How many iterations to evaluate for
    dropout_prob = 0.2
    num_layers = 3
    num_heads = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Using device: ', device)

    data_folder = 'data/FrenchEnglish/'

    # First we read in the data
    # English data
    char_dict_en, data_en = read_in_data(data_folder + 'text_en_lite')
    # French data
    char_dict_fr, data_fr = read_in_data(data_folder + 'text_fr_lite')

    # The sequence lines are of different lengths, so we need to pad them to the same length.
    # We will pad them to the length of the longest sequence
    # Split the sequences over the new line character
    data_en = data_en.split('\n')
    data_fr = data_fr.split('\n')

    # Find the length of the longest sequence
    max_seq_len_en = max([len(seq) for seq in data_en])
    max_seq_len_fr = max([len(seq) for seq in data_fr])
    max_seq_len = max(max_seq_len_en, max_seq_len_fr)

    # Print the first 10 sequences
    print('First 10 English sequences: ', data_en[:10])
    print('First 10 French sequences: ', data_fr[:10])

    # Create the encoder and decoder dictionaries and the encode and decode functions
    encoder_dict_en, decoder_dict_en, encode_en, decode_en = create_simple_encoder_decoder(char_dict_en)

    encoder_dict_fr, decoder_dict_fr, encode_fr, decode_fr = create_simple_encoder_decoder(char_dict_fr)

    # Encode the data
    data_en_encoded = [encode_en(seq) for seq in data_en]
    data_fr_encoded = [encode_fr(seq) for seq in data_fr]

    # Add <sos>, <eos> and <pad> tokens to the data
    # <pad> token is 0, <sos> token is 1 and <eos> token is 2

    data_en_encoded = [[1] + seq + [2] + [0] * (max_seq_len - len(seq)) for seq in data_en_encoded]
    data_fr_encoded = [[1] + seq + [2] + [0] * (max_seq_len - len(seq)) for seq in data_fr_encoded]

    # Print the encoded data
    # print('Encoded English data: ', data_en_encoded[:10])
    # print('Encoded French data: ', data_fr_encoded[:10])

    # Update the max sequence length to include the <sos> and <eos> tokens
    max_seq_len += 2

    # Save the encoded data
    # save_data_as_txt(data_en_encoded, data_folder + 'text_en_encoded')
    # save_data_as_txt(data_fr_encoded, data_folder + 'text_fr_encoded')

    # Train the model
    # Create the model
    transformer_hyperparams = {'src_vocab_size': len(encoder_dict_en.keys()),
                               'trg_vocab_size': len(encoder_dict_fr.keys()),
                               'embedding_dim': embedding_dim,
                               'num_layers': num_layers,
                               'num_heads': num_heads,
                               'dropout_prob': dropout_prob,
                               'max_seq_length': max_seq_len}

    model = Transformer(**transformer_hyperparams).to(device)

    # Create the loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # Create the optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # Convert each sequence to a tensor
    data_en_encoded = [torch.tensor(seq, dtype=torch.long, device=device) for seq in data_en_encoded]
    data_fr_encoded = [torch.tensor(seq, dtype=torch.long, device=device) for seq in data_fr_encoded]

    # Create the train and evaluation data
    train_data = list(zip(data_en_encoded, data_fr_encoded))
    random.shuffle(train_data)
    train_data = train_data[:int(len(train_data) * 0.8)]
    eval_data = train_data[int(len(train_data) * 0.8):]

    data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                              shuffle=True)

    eval_data_loader = torch.utils.data.DataLoader(eval_data,
                                                   batch_size=1,
                                                   shuffle=True)



    # data_loader = torch.utils.data.DataLoader(list(zip(data_en_encoded, data_fr_encoded)), batch_size=batch_size,
    #                                           shuffle=True)
    #
    # eval_data_loader = torch.utils.data.DataLoader(list(zip(data_en_encoded, data_fr_encoded)),
    #                                                batch_size=batch_size,
    #                                                shuffle=True)
    #

    # Split the data into training and evaluation sets


    # Create the loss history
    loss_history = []
    eval_loss_history = []
    # Train the model
    start_time = time.time()
    last_time = start_time
    for i in range(max_iters):
        # Get the next batch
        batch = next(iter(data_loader))
        # Unpack the batch
        x, y = batch

        # Get the predictions
        y_pred = model(x, y)
        # Calculate the loss

        loss = loss_fn(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        # Back propagate the loss
        loss.backward()
        # Update the weights
        optimiser.step()
        # Reset the gradients
        optimiser.zero_grad()
        # Add the loss to the loss history
        loss_history.append(loss.item())
        # Print the loss
        if i % 10 == 0:
            print('Iteration: ', i, ' Loss: ', loss.item())
        # Evaluate the model
        if i % eval_every == 0:
            # Set the model to evaluation mode
            model.eval()

            # Create the loss history
            eval_loss_history = []
            # Evaluate the model
            for j in range(eval_iters):
                # Get the next batch
                batch = next(iter(eval_data_loader))
                # Unpack the batch
                x, y = batch

                # Get the prediction - here model is in inference mode so we don't need to pass the target sequence
                y_pred = model(x) #.translate(x)
                # Calculate the loss
                loss = loss_fn(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
                # Add the loss to the loss history
                eval_loss_history.append(loss.item())
            # Print the loss
            print('Evaluation loss: ', sum(eval_loss_history) / len(eval_loss_history))
            # Save the model every eval_every iterations

        if i % eval_every == 0 and i > 0:
            # torch.save(model.state_dict(), 'models/transformer_model')
            model.save(f"saved_models/{type(model).__name__}_{i}_iters.pt")
            # Calculate the time taken for the last eval_every iterations
            current_time = time.time()
            print('Time taken for last', eval_every, 'iterations: ', current_time - last_time)
            last_time = current_time

        # Set the model back to training mode

        model.train()




    # Plot the loss history
    plt.plot(loss_history, label='Training loss')
    plt.plot(eval_loss_history, label='Evaluation loss')
    plt.legend()
    plt.show()

    # Generate some translations
    # Set the model to evaluation mode
    model.eval()
    # Generate some translations
    for i in range(10):
        # Get a random sequence in English and its translation in French and compare with model output
        samp_index = random.randint(0, len(data_en_encoded) - 1)
        # Get the English sequence
        x = data_en_encoded[samp_index]
        # Get the French sequence
        y = data_fr_encoded[samp_index]
        # Get the model output
        y_pred = model(x, y)
        # Get the predicted sequence
        y_pred = torch.argmax(y_pred, dim=-1)
        # Print the English sequence
        print('English: ', decode_en(x.tolist()))
        # Print the French sequence
        print('French: ', decode_fr(y.tolist()))
        # Print the predicted sequence
        print('Predicted: ', decode_fr(y_pred.tolist()))
        print('')
