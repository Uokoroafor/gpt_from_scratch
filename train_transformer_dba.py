# Test file for training full transformer model on a translation task
# Imports
import random
import torch
from my_models.my_transformer import Transformer
from utils.basic_tokeniser import create_simple_encoder_decoder
from utils.data_utils import read_in_data

if __name__ == '__main__':
    torch.manual_seed(6345789)  # Set the random seed for reproducibility
    # Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

    # Set Hyperparameters
    batch_size = 32  # This is the size of the batch of data that will be processed at once
    block_size = 64  # This is the size of the context window
    max_iters = 200  # How many iterations to train for
    eval_every = max_iters // 10  # How often to evaluate the model
    embedding_dim = 256  # The size of the embedding dimension
    lr = 3e-4
    eval_iters = 100  # How many iterations to evaluate for
    dropout_prob = 0.2
    num_layers = 2
    num_heads = 4

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
    #
    # # Create the loss function
    # loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    #
    # # Create the optimiser
    # optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    #
    # # Convert each sequence to a tensor
    data_en_encoded = [torch.tensor(seq, dtype=torch.long, device=device) for seq in data_en_encoded]
    data_fr_encoded = [torch.tensor(seq, dtype=torch.long, device=device) for seq in data_fr_encoded]
    #
    # # Create the data loader
    # data_loader = torch.utils.data.DataLoader(list(zip(data_en_encoded, data_fr_encoded)),
    #                                           batch_size=batch_size,
    #                                           shuffle=True)
    #
    # # Create the loss history
    # loss_history = []
    # eval_loss_history = []
    # # Train the model
    # for i in range(max_iters):
    #     # Get the next batch
    #     batch = next(iter(data_loader))
    #     # Unpack the batch
    #     x, y = batch
    #
    #     # Get the predictions
    #     y_pred = model(x, y)
    #     # Calculate the loss
    #
    #     loss = loss_fn(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
    #     # Back propagate the loss
    #     loss.backward()
    #     # Update the weights
    #     optimiser.step()
    #     # Reset the gradients
    #     optimiser.zero_grad()
    #     # Add the loss to the loss history
    #     loss_history.append(loss.item())
    #     # Print the loss
    #     if i % 10 == 0:
    #         print('Iteration: ', i, ' Loss: ', loss.item())
    #     # Evaluate the model
    #     if i % eval_every == 0:
    #         # Set the model to evaluation mode
    #         model.eval()
    #         # Create the evaluation data loader
    #         eval_data_loader = torch.utils.data.DataLoader(list(zip(data_en_encoded, data_fr_encoded)),
    #                                                        batch_size=batch_size,
    #                                                        shuffle=True)
    #         # Create the loss history
    #         eval_loss_history = []
    #         # Evaluate the model
    #         for j in range(eval_iters):
    #             # Get the next batch
    #             batch = next(iter(eval_data_loader))
    #             # Unpack the batch
    #             x, y = batch
    #             # Get the predictions
    #             y_pred = model(x, y)
    #             # Calculate the loss
    #             loss = loss_fn(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
    #             # Add the loss to the loss history
    #             eval_loss_history.append(loss.item())
    #         # Print the loss
    #         print('Evaluation loss: ', sum(eval_loss_history) / len(eval_loss_history))
    #         # Save the model every eval_every iterations
    #     if i % eval_every == 0:
    #         # torch.save(model.state_dict(), 'models/transformer_model')
    #         model.save(f"saved_models/{type(model).__name__}_{i}_iters.pt")
    #
    #     # Set the model back to training mode
    #
    #     model.train()
    #
    # # Plot the loss history
    # plt.plot(loss_history, label='Training loss')
    # plt.plot(eval_loss_history, label='Evaluation loss')
    # plt.legend()
    # plt.show()

    # Load the model state dict from file
    path = "saved_models/Transformer_200_iters_230523-0037.pt"
    # model = Transformer(**transformer_hyperparams).to(device)
    model.load_state_dict(torch.load(path))
    # Generate some translations
    # Set the model to evaluation mode
    model.eval()
    # Generate some translations
    # 10 random French sentences:
    fr_seq = list('je suis tres froid.')
    en_seq = list('i am very cold.')



    # Encode the French sentence
    fr_seq_encoded = encode_fr(fr_seq)
    # Encode the English sentence
    en_seq_encoded = encode_en(en_seq)

    en_seq_encoded = [[1] + en_seq_encoded + [2] + [0] * (max_seq_len-2 - len(en_seq))]
    fr_seq_encoded = [[1] + fr_seq_encoded + [2] + [0] * (max_seq_len-2 - len(fr_seq))]
    # Convert each sequence to a tensor
    en_seq_encoded = torch.tensor(en_seq_encoded, dtype=torch.long, device=device)
    fr_seq_encoded = torch.tensor(fr_seq_encoded, dtype=torch.long, device=device)


    for i in range(10):
        # Get a random sequence in English and its translation in French and compare with model output
        samp_index = random.randint(0, len(data_en_encoded) - 1)
        # Get the English sequence
        x = data_en_encoded[samp_index]
        # Get the French sequence
        y = data_fr_encoded[samp_index]
        # print('Sample index: ', samp_index)
        # print('English: ', decode_en(x.tolist()))
        # print('French: ', decode_fr(y.tolist()))

        # copy x and y to x_ and y_
        x_ = x.clone()
        y_ = y.clone()


        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        print('x.shape, y.shape:', x.shape, y.shape)


        # Get the model output
        y_pred = model(x)
        # Get the predicted sequence
        y_pred = torch.argmax(y_pred, dim=-1)
        print('y_pred.shape:', y_pred.shape)
        # Truncate the sequence to the first EOS token
        x_ = x_[x_ != 0]
        y_ = y_[y_ != 0]
        y_pred = y_pred[0, :len(y_)]

        # Print the English sequence
        print('En: ', ''.join(decode_en(x_.tolist())))
        # Print the French sequence
        print('Fr: ', ''.join(decode_fr(y_.tolist())))
        # Print the predicted sequence
        print('Pr: ', ''.join(decode_fr(y_pred.tolist())))
        print('')

    # Test the encoded sentences
    # print('En: ', ''.join(decode_en(en_seq_encoded.tolist())))
    # print('Fr: ', ''.join(decode_fr(fr_seq_encoded.tolist())))
    # # Generate the translation
    y_pred = model(en_seq_encoded)
    # Get the predicted sequence
    y_pred = torch.argmax(y_pred, dim=-1)
    # Truncate the sequence to the first EOS token
    y_pred = y_pred[0, :torch.sum(fr_seq_encoded != 0)]
    # Print the predicted sequence
    print('Pr: ', ''.join(decode_fr(y_pred.tolist())))
    print('')
