# Create a large language model from scratch, using pytorch.

# Import the necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
import os

# Set the random seed manually for reproducibility.
torch.manual_seed(1111)

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Define the hyperparameters
n_epochs = 2000
print_every = 100
plot_every = 10
hidden_size = 256
n_layers = 2
lr = 0.001

# Read in the data
with open('data/input.txt', 'r') as f:
    text = f.read()
