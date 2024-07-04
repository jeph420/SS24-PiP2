import torch
from torch.utils.data import random_split, TensorDataset


def get_dataset():
    torch.random.manual_seed(0)
    num_train_examples = 10000
    training_data = torch.rand(num_train_examples, 32)
    random_function = torch.rand(32)
    sums = torch.sum(training_data * random_function, dim=1) ** 2
    targets = torch.where(sums < 100, sums, torch.zeros_like(sums))
    all_data = TensorDataset(training_data, targets)
    training_data, eval_data = random_split(all_data, [int(num_train_examples * 0.75), int(num_train_examples * 0.25)])
    return training_data, eval_data
