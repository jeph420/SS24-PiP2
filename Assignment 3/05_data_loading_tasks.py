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

import numpy as np
import torch

#
# Task 1
#

# Write a function "sequence_collate_fn(batch_as_list)" that stacks samples
# containing sequences of varying lengths into minibatches. "batch_as_list" is a
# list of samples, where each sample is a tuple (sequence_array, label).
# "sequence_array" is a numpy array of shape (seq_length, n_features=3), where
# "seq_length" can vary for each sequence but is >= 1. "label" is a numpy array
# of shape "(2, 2)". The minibatch entries for "sequence_array" should be
# stacked in the first dimension, padding the sequence at the sequence ends with
# 0-values to the same length. The minibatch entries for "label" should be
# stacked in the first dimension. Your function should return a tuple
# (stacked_sequences, stacked_labels), where both tuple entries are PyTorch
# tensors of datatype torch.float32.
#
# Hint: Get the maximum sequence length within the current minibatch and create
# a tensor that contains only 0 values and can hold all stacked sequences. Then,
# write the sequences into the tensor.

rng = np.random.default_rng(seed=0)
batch_as_list_1 = [(rng.uniform(size=(rng.integers(low=1, high=10), 3)),
                    rng.uniform(size=(2, 2))) for _ in range(4)]  # mb_size 4
batch_as_list_2 = [(rng.uniform(size=(rng.integers(low=1, high=10), 3)),
                    rng.uniform(size=(2, 2))) for _ in range(3)]  # mb_size 3


def sequence_collate_fn(batch_as_list: list):
    # Your code here #
    pass


#
# Task 2
#

# Write a function "one_hot_collate_fn(batch_as_list)" that stacks samples
# containing one-hot features into minibatches. "batch_as_list" is a list of
# samples, where each sample is a tuple (one_hot_feat, label). "one_hot_feat" is
# a numpy array of shape (n_features=3,), containing only the indices of the
# 1-entries in the one-hot feature vector. The dictionary size is given with
# DICT_SIZE=11, i.e., each full one-hot encoded feature vector ultimately has a
# length of 11 (where one entry is 1). The full one-hot feature matrix thus
# should have shape (3, DICT_SIZE=11). "label" is a numpy array of shape (2, 2).
# The minibatch entries for "one_hot_feat" should be stacked in the first
# dimension as full one-hot feature vectors. The minibatch entries for "label"
# should be stacked in the first dimension. Your function should return a tuple
# (stacked_sequences, stacked_labels), where both tuple entries are PyTorch
# tensors of datatype torch.float32.
#
# Hint: First allocate a tensor filled with 0-values that can fit all stacked
# full one-hot feature vectors of the minibatch. Then use "one_hot_feat" as
# indices to set elements to 1.

DICT_SIZE = 11
rng = np.random.default_rng(seed=0)
batch_as_list_1 = [(rng.integers(low=0, high=DICT_SIZE, size=(3,)),
                    rng.uniform(size=(2, 2))) for _ in range(4)]  # mb_size 4
batch_as_list_2 = [(rng.integers(low=0, high=DICT_SIZE, size=(3,)),
                    rng.uniform(size=(2, 2))) for _ in range(3)]  # mb_size 3


def one_hot_collate_fn(batch_as_list: list):
    # Your code here #
    pass


#
# Task 3
#

# Same as task 2, but now send only the indices of the one-hot feature vector to
# the GPU and create the stacked full one-hot feature vector on the GPU.
#
# Hint: You can send a tensor to the GPU by specifying "device='cuda:0'". If you
# do not have a GPU, set device to 'cpu'.

DICT_SIZE = 11
rng = np.random.default_rng(seed=0)
batch_as_list_1 = [(rng.integers(low=0, high=DICT_SIZE, size=(3,)),
                    rng.uniform(size=(2, 2))) for _ in range(4)]  # mb_size 4
batch_as_list_2 = [(rng.integers(low=0, high=DICT_SIZE, size=(3,)),
                    rng.uniform(size=(2, 2))) for _ in range(3)]  # mb_size 3


def one_hot_collate_fn(batch_as_list: list):
    # Your code here #
    pass


#
# Task 4
#

# Create a class "ImageDataset" extending the class "torch.utils.data.Dataset"
# to load all RGB images from a directory specified in __init__. The image files
# should be all files that end with ".jpg", and they should be sorted according
# to their file names (in ascending order). In the __getitem__ method, the image
# for the specified index is then loaded from the disk, transformed into a numpy
# array of datatype float32, scaled to [0, 1] and ultimately normalized given
# the following mean and std arrays (values for each channel):
# norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# This transformed image is then returned, together with the index (=ID of the
# image).
#
# Finally, use an instance from the class "torch.utils.data.DataLoader" to split
# the image set into minibatches (you can choose the settings yourself). Print
# those minibatches afterwards.
#
# Hint: You can use the sample images in "05_images.zip" (unzip first), which is
# a subset of following image dataset:
# https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Your code here #
