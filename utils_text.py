import pdb

import yaml
import termplotlib as tpl
import numpy as np
from hyperpyyaml import load_hyperpyyaml
import random
import torch


RED = '\033[91m'
RESET = '\033[0m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'


def plot_histogram(train_data, bin_count=10, color=BLUE, key='target'):
    print(f"{color}")
    target_lengths = []
    for item in train_data:
        target = item[key]
        target = remove_duplicates(target)
        target_lengths.append(len(target))

    # Compute histogram data
    hist, bins = np.histogram(target_lengths, bins=bin_count)

    # Create and show the plot
    fig = tpl.figure()
    fig.hist(hist, bins, force_ascii=True, orientation="horizontal")
    fig.show()
    print(f"{RESET}")

def load_config(config_path):
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = load_hyperpyyaml(file)
    return config

def remove_duplicates(sequence, remove_consecutive=True):
    numbers = sequence.split()
    if remove_consecutive:
        distinct_numbers = [numbers[i] for i in range(len(numbers)) if i == 0 or numbers[i] != numbers[i - 1]]
        return distinct_numbers
    else:
        return numbers