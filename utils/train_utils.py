import pickle
from typing import Dict, Optional, Tuple
import torch
from torch import nn
from utils.file_utils import create_training_folder, save_losses, save_config
from utils.plot_utils import plot_losses
from utils.time_utils import EpochTimer
from utils.logging_utils import TrainingLogger


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optimiser: torch.optim.Optimizer,
            loss_fn: torch.nn.modules.loss._Loss,
            training_hyperparameters: Dict,
            encoding_utils: Dict,
    ):
        """Constructor class for Trainer
        Args:
            model (nn.Module): Model to train
            optimiser (torch.optim.Optimizer): Optimiser to use for training
            loss_fn (torch.nn.modules.loss._Loss): Loss function to use for training
            training_hyperparameters (Dict): Dictionary containing training hyperparameters
            encoding_utils (Dict): Dictionary containing encoder/decoder dictionaries and functions
        """
        self.train_data = None
        self.val_data = None
        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.encoding_utils = encoding_utils
        self.best_model_dict = None

        # Preallocate variables defined in set_training_hyperparameters
        self.device = None
        self.epochs = None
        self.batch_size = None
        self.eval_every = None
        self.eval_iters = None
        self.max_seq_len = None
        self.save_every = None

        # Create a folder to save the model and training losses
        self.path = create_training_folder()

        # Unpack training hyperparameters
        self._set_training_hyperparameters(**training_hyperparameters)

        # Save the training hyperparameters as a  txt file
        save_config(training_hyperparameters, f"{self.path}/config.txt")

        # Save the model architecture as a txt file
        with open(f"{self.path}/model.txt", "w") as f:
            f.write(str(self.model))

        # Save the parameters of the model as a txt file
        save_config(self.model.count_parameters(), f"{self.path}/model_parameters.txt")

        # Save the encoding_utils as a pickle file
        filename = f"{self.path}/encoding_utils.pkl"
        with open(filename, "wb") as file:
            pickle.dump(self.encoding_utils, file)

    def train(
            self,
            train_data: torch.Tensor,
            val_data: torch.Tensor,
            save_model: bool = True,
            save_model_path: Optional[str] = None,
            plotting: bool = True,
            verbose: Optional[bool] = True,
            early_stopping: bool = False,
    ):
        """Train the model
        Args:
            train_data (torch.Tensor): Training data
            val_data (torch.Tensor): Validation data
            save_model (bool, optional): Whether to save the model(s) and save the best model. Defaults to True.
            save_model_path (Optional[str], optional): Path to save the model. Defaults to None.
            plotting (bool, optional): Whether to plot the losses. Defaults to True.
            verbose (Optional[bool], optional): Whether to print the progress of training. Defaults to True.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.

        """

        self.train_data = train_data
        self.val_data = val_data

        train_losses = []
        val_losses = []
        lowest_val_loss = float("inf")
        logger = TrainingLogger(self.path + "/training_logs/training_log.txt", name="training_log", verbose=verbose)

        logger.log_info(
            f"Training {type(self.model).__name__} for {self.epochs} iterations"
        )

        try:
            timer = EpochTimer()
            timer.start()
            decode = self.encoding_utils["decode_fn"]

            for i in range(self.epochs + 1):
                # Running for one extra epoch to get the final validation loss
                if i % self.eval_every == 0:
                    losses = self._estimate_loss()
                    logger.log_info(
                        f'At Iteration: {max(1, i)}/{self.epochs}, Train loss: {losses["train"]: .4f}, '
                        f'Val loss: {losses["val"]: .4f}')

                    timer.lap()
                    logger.log_info(
                        timer.print_last_epoch_time(label=f"Time taken for last {self.eval_every} iterations: "))
                    if verbose:
                        # Generate a sample from the model
                        chars = decode(
                            self.model.generate(
                                start_token=self.model.trg_sos * torch.ones((1, 1), dtype=torch.long),
                                max_length=30,
                                sampled=False,
                            )[0].tolist()
                        )
                        logger.log_info(f"Generating 30 characters without sampling: {''.join(chars)}")

                    train_losses.append(losses["train"])
                    val_losses.append(losses["val"])

                    # Update the best model state dict and lowest validation loss
                    lowest_val_loss = self.update_best_model_dict(
                        losses["val"], lowest_val_loss
                    )

                    if early_stopping and i > 0 and val_losses[-1] > val_losses[-2]:
                        logger.log_info(
                            f"Stopping early after {i} iterations")
                        break

                if self.save_every is not None and i % self.save_every == 0:
                    self.save_model(
                        f"{self.path}/saved_models/{type(self.model).__name__}_iter_{max(1, i)}.pt")

                if i == self.epochs:
                    break

                # Get a batch of data
                xb, yb = self._get_batch("train")

                # Zero the gradients
                self.optimiser.zero_grad()

                # Get the embeddings and the loss (Forward pass)
                embeds = self.model(trg=xb)

                loss = self.loss_fn(embeds.view(-1, embeds.size(-1)), yb.view(-1))

                # Back propagate the loss (Backward pass)
                loss.backward()

                # Take a step with the optimiser
                self.optimiser.step()

            timer.lap()
            logger.log_info(timer.print_total_time(label="Total time taken: "))

            if plotting:
                plot_save_path = (
                    f"{self.path}/training_logs/{type(self.model).__name__}_losses.png"
                    if save_model
                    else None)

                plot_losses(
                    train_losses,
                    val_losses,
                    model_name=type(self.model).__name__,
                    num_epochs=self.epochs,
                    saved_path=plot_save_path, )

            if save_model:
                # Load and save the best model
                self.model.load_state_dict(self.best_model_dict)
                save_model_path = self.save_best_model(save_model_path)
                logger.log_info(f"Saved best model at: {save_model_path}")

                # Save the losses
                save_losses(train_losses, val_losses, self.path)
                logger.log_info(f"Saved losses at: {self.path}/training_logs/losses.csv")

            else:
                # If we are not saving the model, load the best model
                self.model.load_state_dict(self.best_model_dict)
        except Exception as e:
            logger.log_error(f"Error while training: {str(e)}")
            raise e

        return self.model, train_losses, val_losses

    def evaluate(self, test_data: torch.Tensor, verbose: bool = True, num_iters: Optional[int] = None):
        """Evaluate the model
        Args:
            test_data (torch.Tensor): Test data
            verbose (bool, optional): Whether to print the progress of evaluation. Defaults to True.
            num_iters (Optional[int], optional): Number of iterations to evaluate. Defaults to None
            (Evaluate on the entire test data).
        Returns:
            float: Test loss
        """

        self.model.eval()
        if num_iters is None:
            test_loss = self._calculate_test_loss(test_data)
            if verbose:
                print(f"Test loss: {test_loss: .4f}")
        else:
            test_loss = self._estimate_test_loss(test_data, num_iters=num_iters)
            if verbose:
                print(f"Test loss: {test_loss: .4f}")
        return test_loss

    def _set_training_hyperparameters(self, **kwargs):
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

    def save_best_model(self, best_model_path: Optional[str]):
        """Save the best model
        Args:
            best_model_path (Optional[str]): Path to save the best model
        """
        if best_model_path is None:
            best_model_path = (
                f"{self.path}/saved_models/{type(self.model).__name__}_best.pt"
            )
        self.save_model(best_model_path)
        return best_model_path

    def update_best_model_dict(self, loss_val: float, lowest_val_loss: float) -> float:
        """Update the best model dictionary if the validation loss is the lowest so far
        Args:
            loss_val (float): Dictionary containing the training and validation losses
            lowest_val_loss (float): Lowest validation loss so far
        """
        if loss_val < lowest_val_loss:
            # Update the lowest validation loss
            lowest_val_loss = loss_val
            # Save the model state dict
            self.best_model_dict = self.model.state_dict()
        return lowest_val_loss

    def _get_batch(self, split: Optional[str] = None, data: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Get a batch of data from the train, validation or a provided data tensor
        Args:
            split (Optional[str], optional): Split to get the data from. Defaults to None.
            data (Optional[torch.Tensor], optional): Data tensor to get the batch from. Defaults to None.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Batch of data
        """

        if data is not None:
            data = data
        else:
            if split == "train":
                data = self.train_data
            elif split == "val":
                data = self.val_data
            else:
                raise ValueError(f"Unknown split: '{split}'")
        ix = torch.randint(len(data) - self.max_seq_len, (self.batch_size,))
        x = torch.stack([data[i: i + self.max_seq_len] for i in ix])
        y = torch.stack([data[i + 1: i + self.max_seq_len + 1] for i in ix])
        x, y = x.to(self.device), y.to(
            self.device
        )  # Transfer the data to the GPU if we are using it
        return x, y

    @torch.no_grad()
    def _estimate_loss(self) -> Dict[str, float]:
        """Estimate the loss for the data

        Returns:
            Dict[str, float]: Dictionary containing the training and validation losses
        """
        self.model.eval()  # Set the model to evaluation mode
        out = {}
        for split in ["train", "val"]:
            losses = []
            for i in range(self.eval_iters):
                x, y = self._get_batch(split)
                embeds = self.model(trg=x)
                loss = self.loss_fn(embeds.flatten(end_dim=1), y.flatten())
                losses.append(loss.item())
            out[split] = torch.tensor(losses).mean().item()
        return out

    def _calculate_test_loss(self, test_data: torch.Tensor) -> float:
        """Calculate the loss on the full test data (without sampling)
        Args:
            test_data (torch.Tensor): Test data
        Returns:
            float: Loss on the test data
        """
        self.model.eval()
        test_loss = self.loss_fn(self.model(test_data).view(-1, test_data.size(-1)), test_data.view(-1))
        return test_loss.item()

    def _estimate_test_loss(self, test_data: torch.Tensor, num_iters: int = 100) -> float:
        """Estimate the loss on the test data by sampling a number of batches
        Args:
            test_data (torch.Tensor): Test data
            num_iters (int, optional): Number of samples to estimate the loss. Defaults to 100.
        Returns:
            float: Loss on the test data
        """
        self.model.eval()
        losses = []
        for _ in range(num_iters):
            x, y = self._get_batch(data=test_data)
            embeds = self.model(trg=x)
            loss = self.loss_fn(embeds.flatten(end_dim=1), y.flatten())
            losses.append(loss.item())
        return torch.tensor(losses).mean().item()
