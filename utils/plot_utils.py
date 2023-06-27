from typing import List, Optional
import matplotlib.pyplot as plt
import torch


def plot_losses(train_losses: List[float], val_losses: List[float], model_name: Optional[str] = None, num_epochs: Optional[int] = None):
    """Plot the training and validation losses
    Args:
        train_losses (List[float]): Training losses
        val_losses (List[float]): Validation losses
        model_name (Optional[str], optional): Name of the model. Defaults to None.
        num_epochs (Optional[int], optional): Number of epochs. Defaults to None.
    """
    if num_epochs is not None:
        steps = num_epochs
        x = torch.arange(1, num_epochs + 1, num_epochs//len(train_losses))
    else:
        steps = len(train_losses)
        x = torch.arange(1, len(train_losses) + 1)
    plt.plot(x, train_losses, label='train')
    plt.plot(x, val_losses, label='val')
    if model_name is not None:
        plt.title(f'Losses for the {model_name} model over {steps} iterations')
    else:
        plt.title(f'Losses over {len(train_losses)} steps')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
