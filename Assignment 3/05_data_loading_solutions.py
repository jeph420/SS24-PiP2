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

Example solutions for tasks in the provided tasks file.
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
    #
    # Handle sequences
    #
    # Get sequence entries, which are at index 0 in each sample tuple
    sequences = [sample[0] for sample in batch_as_list]
    # Get the maximum sequence length in the current minibatch
    max_seq_len = np.max([seq.shape[0] for seq in sequences])
    # Allocate a tensor that can fit all padded sequences
    n_seq_features = sequences[0].shape[1]
    stacked_sequences = torch.zeros(size=(len(sequences), max_seq_len, n_seq_features), dtype=torch.float32)
    # Write the sequences into the tensor
    for i, sequence in enumerate(sequences):
        stacked_sequences[i, :len(sequence), :] = torch.from_numpy(sequence)
    
    #
    # Handle labels
    #
    # Get label entries, which are at index 1 in each sample tuple
    labels = [sample[1] for sample in batch_as_list]
    # Convert them to tensors and stack them
    stacked_labels = torch.stack([torch.tensor(label, dtype=torch.float32) for label in labels], dim=0)
    
    return stacked_sequences, stacked_labels


print(f"batch_as_list_1:")
print(f"original shapes: {[(t[0].shape, t[1].shape) for t in batch_as_list_1]}")
# print(f"stacked: {sequence_collate_fn(batch_as_list_1)}")
print(f"stacked shapes: {[t.shape for t in sequence_collate_fn(batch_as_list_1)]}")
print(f"batch_as_list_2:")
print(f"original shapes: {[(t[0].shape, t[1].shape) for t in batch_as_list_2]}")
# print(f"{sequence_collate_fn(batch_as_list_2)}")
print(f"stacked shapes: {[t.shape for t in sequence_collate_fn(batch_as_list_2)]}")


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
    #
    # Handle one-hot features
    #
    # Get one-hot feature entries, which are at index 0 in each sample tuple
    one_hot_indices = [sample[0] for sample in batch_as_list]
    # Allocate a tensor that can fit the stacked full one-hot features
    n_one_hot_features = one_hot_indices[0].shape[0]
    stacked_one_hot_features = torch.zeros(size=(len(one_hot_indices), n_one_hot_features, DICT_SIZE),
                                           dtype=torch.float32)
    
    # Write the indices into the tensor
    feature_index = torch.arange(n_one_hot_features)
    for i, one_hot_inds in enumerate(one_hot_indices):
        stacked_one_hot_features[i, feature_index, one_hot_inds] = 1
    
    #
    # Handle labels
    #
    # Get label entries, which are at index 1 in each sample tuple
    labels = [sample[1] for sample in batch_as_list]
    # Convert them to tensors and stack them
    stacked_labels = torch.stack([torch.tensor(label, dtype=torch.float32) for label in labels], dim=0)
    
    return stacked_one_hot_features, stacked_labels


print(f"batch_as_list_1:")
print(f"original: {batch_as_list_1}")
print(f"{one_hot_collate_fn(batch_as_list_1)}")
print(f"stacked shapes: {[t.shape for t in one_hot_collate_fn(batch_as_list_1)]}")
print(f"batch_as_list_2:")
print(f"original: {batch_as_list_2}")
print(f"{one_hot_collate_fn(batch_as_list_2)}")
print(f"stacked shapes: {[t.shape for t in one_hot_collate_fn(batch_as_list_2)]}")


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
    device = "cpu"  # Use "cuda" if you have a GPU, otherwise "cpu"
    
    #
    # Handle one-hot features
    #
    # Get one-hot feature entries, which are at index 0 in each sample tuple
    one_hot_indices = [sample[0] for sample in batch_as_list]
    # Allocate a tensor on GPU that can fit the stacked full one-hot features
    n_one_hot_features = one_hot_indices[0].shape[0]
    stacked_one_hot_features = torch.zeros(size=(len(one_hot_indices), n_one_hot_features, DICT_SIZE),
                                           dtype=torch.float32, device=device)
    
    # We need the one-hot indices on the GPU. We could send them ony by one, but
    # packing them into one large tensor before sending them will be faster.
    one_hot_indices = torch.stack([torch.tensor(i) for i in one_hot_indices])
    one_hot_indices = one_hot_indices.to(dtype=torch.long, device=device)
    # Write the indices into the tensor (ensure all the tensors are on the GPU)
    feature_index = torch.arange(n_one_hot_features, device=device)
    for i, one_hot_inds in enumerate(one_hot_indices):
        stacked_one_hot_features[i, feature_index, one_hot_inds] = 1
    
    #
    # Handle labels
    #
    # Get label entries, which are at index 1 in each sample tuple
    labels = [sample[1] for sample in batch_as_list]
    # Convert them to tensors and stack them
    stacked_labels = torch.stack([torch.tensor(label, dtype=torch.float32) for label in labels], dim=0)
    
    return stacked_one_hot_features, stacked_labels


# Note: For small arrays, the trade-off between GPU bandwidth vs. computation
# time on GPU will not justify this version. However, if you are working with
# large data, these tricks can make your code executable in reasonable time (see
# benchmark code below).
print(f"batch_as_list_1:")
print(f"original: {batch_as_list_1}")
print(f"{one_hot_collate_fn(batch_as_list_1)}")
print(f"stacked shapes: {[t.shape for t in one_hot_collate_fn(batch_as_list_1)]}")
print(f"batch_as_list_2:")
print(f"original: {batch_as_list_2}")
print(f"{one_hot_collate_fn(batch_as_list_2)}")
print(f"stacked shapes: {[t.shape for t in one_hot_collate_fn(batch_as_list_2)]}")


#
# Benchmarking against "simple" implementation without sending indices only
#

def simple_one_hot_collate_fn(batch_as_list: list):
    # Create full one-hot array on CPU, send full one-hot array to GPU.
    device = "cpu"  # Use "cuda" if you have a GPU, otherwise "cpu"
    
    #
    # Handle one-hot features
    #
    # Get one-hot feature entries, which are at index 0 in each sample tuple
    one_hot_indices = [sample[0] for sample in batch_as_list]
    # Allocate a tensor on GPU that can fit the stacked full one-hot features
    n_one_hot_features = one_hot_indices[0].shape[0]
    stacked_one_hot_features = torch.zeros(size=(len(one_hot_indices), n_one_hot_features, DICT_SIZE),
                                           dtype=torch.float32)
    
    # Write the indices into the tensor
    feature_index = torch.arange(n_one_hot_features)
    for i, one_hot_inds in enumerate(one_hot_indices):
        stacked_one_hot_features[i, feature_index, one_hot_inds] = 1
    stacked_one_hot_features = stacked_one_hot_features.to(device=device)
    
    #
    # Handle labels
    #
    # Get label entries, which are at index 1 in each sample tuple
    labels = [sample[1] for sample in batch_as_list]
    # Convert them to tensors and stack them
    stacked_labels = torch.stack([torch.tensor(label, dtype=torch.float32) for label in labels], dim=0)
    
    return stacked_one_hot_features, stacked_labels


# Measure computation times
import timeit

setup_code = "from __main__ import one_hot_collate_fn, batch_as_list_1, batch_as_list_2"
test_code = "(one_hot_collate_fn(batch_as_list_1), one_hot_collate_fn(batch_as_list_2))"
inds = timeit.timeit(test_code, number=100, setup=setup_code)
setup_code = "from __main__ import simple_one_hot_collate_fn, batch_as_list_1, batch_as_list_2"
test_code = "(simple_one_hot_collate_fn(batch_as_list_1), simple_one_hot_collate_fn(batch_as_list_2))"
simple = timeit.timeit(test_code, number=100, setup=setup_code)

# "inds" version should be faster at DICT_SIZE = 110000
print(f"Sending indices: {inds}; Sending whole array: {simple}")


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


class ImageDataset(Dataset):
    
    def __init__(self, image_dir):
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))
        # Mean and std arrays could also be defined as class attributes
        self.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __getitem__(self, index):
        # Open image file, convert to numpy array and scale to [0, 1]
        with Image.open(self.image_files[index]) as im:
            image = np.array(im, dtype=np.float32) / 255
        # Perform normalization for each channel
        image = (image - self.norm_mean) / self.norm_std
        return image, index
    
    def __len__(self):
        return len(self.image_files)


image_dataset = ImageDataset("05_images")
image_loader = DataLoader(image_dataset, shuffle=True, batch_size=10)

# Iterate through the dataloader
for batch, (images, ids) in enumerate(image_loader):
    print(f"Batch {batch}:")
    print(f"image ids: {ids}")
    print(f"batch shape: {images.shape}")
