import os
import time
from typing import List, Optional


def create_training_folder(path: Optional[str] = None) -> str:
    """Create a training folder to save the model checkpoints and training logs. Folder name is today's date and time.
    Args:
        path (Optional[str], optional): The path to save the training folder to. Defaults to None.
    Returns:
        str: The path to the training folder.
    """

    # Get today's date and time
    # Want the date in the format YY-MM-DD-HH-MM
    date_time = time.strftime('%y%m%d-%H%M')

    # If the training folder doesn't exist, create it
    if not os.path.exists('training'):
        os.makedirs('training')

    # Create the path to the training folder
    if path is None:
        path = f'training/{date_time}'
    else:
        path = f'{path}/{date_time}'

    # Create the training folder
    os.makedirs(path)

    # Also make a folder for the saved models and training logs
    os.makedirs(f'{path}/saved_models')
    os.makedirs(f'{path}/training_logs')

    return path


def save_losses(train_losses: List[float], val_losses: List[float], path: str) -> None:
    """Save the training and validation losses to a txt file.
    Args:
        train_losses (List[float]): Training losses.
        val_losses (List[float]): Validation losses.
        path (str): The path to save the losses to.
    """
    # Save the training and validation losses to a csv file
    with open(f'{path}/training_logs/losses.csv', 'w') as f:
        f.write('train_loss,val_loss\n')
        for train_loss, val_loss in zip(train_losses, val_losses):
            f.write(f'{train_loss},{val_loss}\n')
