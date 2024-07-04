# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

Tasks for self-study. Try to solve these tasks on your own and compare your
solutions to the provided solutions file.
"""

import matplotlib.pyplot as plt
import torch

torch.manual_seed(0)  # Set a known random seed for reproducibility


#
# Task 1
#

# Implement gradient descent using the gradients computed via "backward()"
# without using the PyTorch optimizer. The goal is to get the loss below 0.1 by
# optimizing the trainable weight tensor "weights". The loss should be computed
# from the output and the target "target_tensor". The output calculation should
# be done using
#   output = input_tensor.matmul(weights)
# and the loss should be the mean-squared error
#   loss = ((target_tensor - output) ** 2).mean()
# The "input_tensor" and "target_tensor" can be obtained by iterating over the
# "sample_generator()" generator:
#   for update, (input_tensor, target_tensor) in enumerate(sample_generator()):
#       CODE
#
# Hint: For gradient descent, you will need to compute the gradients of the loss
# w.r.t. the trainable weight tensor. You then multiply the negative gradients
# with a learning rate that is in range [0, 1], which gives you the update you
# have to apply to the trainable weight tensor. Adding the update to the
# trainable weight tensor is one update step. This update however, should NOT be
# recorded by autograd, so you will have to disable the gradient calculation
# when performing the update step. This can be done with the "torch.no_grad()"
# context manager. Lastly, do not forget to zero the gradients after calling
# "backward()". More information on gradient descent (and a nice analogy for
# understanding): https://en.wikipedia.org/wiki/Gradient_descent

def sample_generator():
    """Function returning a generator to generate random samples"""
    while True:
        input_tensor = torch.rand(size=(7,), dtype=torch.float32)
        target_tensor = torch.stack([input_tensor.sum(), input_tensor.sum() * 2])
        yield input_tensor, target_tensor


# The trainable weight tensor we want to optimize
weights = torch.nn.Parameter(torch.rand(size=(7, 2), dtype=torch.float32))

# Your code here #


#
# Task 2
#

# Using the same "sample_generator()" from above, train a simple model of your
# choice (e.g., just a linear layer) using stochastic gradient descent and the
# mean-squared error as loss function for a fixed number of iterations/updates.
# The hyperparameter settings are up to you.

# Your code here #


#
# Task 3
#

# Given 2D samples "data" and their corresponding class labels "targets", create
# a PyTorch Dataset and a DataLoader, and then train a model of your choice.
# Afterward, get the model's predictions for the same data. Create a plot that
# shows the loss values collected during training, a scatter plot of the data
# colored according to the ground truth ("targets") and a scatter plot of the
# data colored according to the model's predictions. Also, include the accuracy
# in this last plot. All hyperparameters (optimizer, learning rate, batch size,
# number of epochs, etc.) are up to you.
#
# Hint: Choose a fitting loss function (classification task). For getting the
# predictions, you need to inspect the output of your model and select the one
# with the highest value (intuition: the model output are probabilities for each
# class). The index of this highest value entry is then the predicted class (you
# can use the tensor method "argmax" to obtain this index). The accuracy is
# defined as the fraction of correct predictions, i.e., you count how often the
# model's predicted class is equal to the true target class, and then you divide
# by the total number of samples.

# Each data_blobs line represents one class with 2D samples in the shape (x, y)
# and class i, where "x" and "y" are drawn from a normal distribution with the
# specified mean and standard deviation, and "i" is ranging from 0 to
# len(data_blobs)
data_blobs = [
    {"n": 50, "m": (1.0, 2.0), "s": 0.5},  # n = number of samples
    {"n": 50, "m": (3.0, 2.5), "s": 0.3},  # m = sample mean (for x and y)
    {"n": 50, "m": (2.0, 1.0), "s": 0.3},  # s = sample standard deviation
]
data = []
targets = []
for i, data_blob in enumerate(data_blobs):
    x = torch.normal(mean=data_blob["m"][0], std=data_blob["s"], size=(data_blob["n"], 1))
    y = torch.normal(mean=data_blob["m"][1], std=data_blob["s"], size=(data_blob["n"], 1))
    target = torch.full(size=(data_blob["n"],), fill_value=i, dtype=torch.long)
    data.append(torch.concat([x, y], dim=1))
    targets.append(target)
data = torch.concat(data, dim=0)
targets = torch.concat(targets, dim=0)

# Your code here #
