import time
from typing import Dict, Optional, Tuple
import torch
from torch import nn
from utils.plot_utils import plot_losses
from utils.file_utils import create_training_folder, save_losses, save_config


# TODO: Add early stopping
# TODO: Add evaluate method
class Trainer:

    def __init__(self, model: nn.Module, optimiser: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss,
                 training_hyperparameters: Dict,
                 device: Optional[str] = 'cpu'):
        """Constructor class for Trainer
        Args:
            model (nn.Module): Model to train
            optimiser (torch.optim.Optimizer): Optimiser to use for training
            loss_fn (torch.nn.modules.loss._Loss): Loss function to use for training
            training_hyperparameters (Dict): Dictionary containing training hyperparameters
            device (Optional[str], optional): Device to use for training. Defaults to 'cpu'.
        """
        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = None
        self.batch_size = None
        self.eval_every = None
        self.eval_iters = None
        self.max_seq_len = None
        self.save_every = None

        # Create a folder to save the model and training losses
        self.path = create_training_folder()

        # Unpack training hyperparameters
        self.set_training_hyperparameters(**training_hyperparameters)

        # Save the training hyperparameters as a  txt file
        save_config(training_hyperparameters, f'{self.path}/config.txt')

        # training_hyperparams = {
        #     'batch_size': 32,
        #     'max_seq_len': 10000,
        #     'epochs': 1000,
        #     'eval_every': 100,
        # }

    def train(self, train_data: torch.Tensor, val_data: torch.Tensor, save_model: bool = True,
              save_model_path: Optional[str] = None, plotting: bool = True, verbose: Optional[bool] = True):
        """Train the model
        Args:
            train_data (torch.Tensor): Training data
            val_data (torch.Tensor): Validation data
            save_model (bool, optional): Whether to save the model. Defaults to True.
            save_model_path (Optional[str], optional): Path to save the model. Defaults to None.
            plotting (bool, optional): Whether to plot the losses. Defaults to True.
            verbose (Optional[bool], optional): Whether to print the progress of training. Defaults to True.
        """

        # Helper functions

        def _get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
            """Get a batch of data from the train, validation or test set.
            Args:
                split: The split to get the batch from.
            Returns:
                A tuple of tensors containing the input and target data.
            """

            if split == 'train':
                data = train_data
            elif split == 'val':
                data = val_data
            else:
                raise ValueError(f"Unknown split: '{split}'")
            ix = torch.randint(len(data) - self.max_seq_len, (self.batch_size,))
            x = torch.stack([data[i:i + self.max_seq_len] for i in ix])
            y = torch.stack([data[i + 1:i + self.max_seq_len + 1] for i in ix])
            x, y = x.to(self.device), y.to(self.device)  # Transfer the data to the GPU if we are using it
            return x, y

        @torch.no_grad()
        def _estimate_loss() -> Dict[str, float]:
            """Estimate the loss for the data

            Returns:
                Dict[str, float]: Dictionary containing the training and validation losses
            """
            self.model.eval()  # Set the model to evaluation mode
            out = {}
            for split in ['train', 'val']:
                losses = torch.zeros(self.eval_iters)
                for i in range(self.eval_iters):
                    x, y = _get_batch(split)
                    embeds = self.model(trg=x)
                    loss = self.loss_fn(embeds.flatten(end_dim=1), y.flatten())
                    losses[i] = loss.item()
                out[split] = losses.mean().item()
            self.model.train()  # Set the model back to training mode
            return out

        train_losses = []
        val_losses = []

        if verbose:
            print(f"Training {type(self.model).__name__} for {self.epochs} iterations...")

        # Measure the time taken for the training
        start_time = time.time()
        last_time = start_time
        for i in range(self.epochs):
            if i % self.eval_every == 0:
                losses = _estimate_loss()
                # Print Step, train loss and validation loss
                if verbose:
                    print(
                        f'At Iteration: {max(1, i)}/{self.epochs}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')
                    print(f'Time taken for last {self.eval_every} iterations: {(time.time() - last_time):.2f} seconds')
                    last_time = time.time()
                train_losses.append(losses["train"])
                val_losses.append(losses["val"])

            if self.save_every is not None and i % self.save_every == 0:
                self.save_model(f'{self.path}/saved_models/{type(self.model).__name__}_iter_{max(1, i)}.pt')

            # Get a batch of data
            xb, yb = _get_batch('train')

            # Zero the gradients
            self.optimiser.zero_grad()

            # Get the embeddings and the loss (Forward pass)
            embeds = self.model(trg=xb)

            loss = self.loss_fn(embeds.view(-1, embeds.size(-1)), yb.view(-1))

            # Backpropagate the loss (Backward pass)
            loss.backward()

            # Take a step with the optimiser
            self.optimiser.step()

        if verbose:
            total_time = int(time.time() - start_time)

            hours = total_time // 3600
            minutes = (total_time % 3600) // 60
            seconds = total_time % 60
            print(f'Time taken for training: {hours} hours, {minutes} minutes, {seconds} seconds')

        if plotting:
            saved_path = f'{self.path}/training_logs/{type(self.model).__name__}_losses.png' if save_model else None
            plot_losses(train_losses, val_losses, model_name=type(self.model).__name__, num_epochs=self.epochs,
                        saved_path=saved_path)

        if save_model:
            if save_model_path is None:
                save_model_path = f'{self.path}/saved_models/{type(self.model).__name__}_final.pt'
            self.save_model(save_model_path)
            save_losses(train_losses, val_losses, self.path)
            if verbose:
                print('Model saved at:', save_model_path)

        return self.model, train_losses, val_losses

    def set_training_hyperparameters(self, **kwargs):
        """Set training hyperparameters
        Args:
            **kwargs: Training hyperparameters
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save_model(self, model_path: str):
        """Save the model
        Args:
            model_path (str): Path to save the model
        """
        torch.save(self.model, model_path)
