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

In this file, we will learn how to use PyTorch to read our dataset.
"""

################################################################################
# PyTorch data handling - Introduction
################################################################################

import numpy as np
import torch

# Numpy random number generation: Legacy and backward compatibility (np < 1.17):
# https://numpy.org/doc/stable/reference/random/legacy.html
# New numpy features (np >= 1.17):
# https://numpy.org/doc/stable/reference/random/index.html

# Setting a seed (https://en.wikipedia.org/wiki/Random_seed) is important to
# get reproducible random numbers.
# Legacy:
np.random.seed(0)
a = np.random.uniform(size=(3,))  # Should be [0.5488135  0.71518937 0.60276338]
print(a)
# New versions:
rng = np.random.default_rng(seed=0)
a = rng.uniform(size=(3,))  # Should be [0.63696169 0.26978671 0.04097352]
print(a)

# Analogous for PyTorch (https://pytorch.org/docs/stable/notes/randomness.html),
# although complete reproducibility is not guaranteed across different releases.
torch.random.manual_seed(0)
a = torch.rand(size=(3,))  # Should be tensor([0.4963, 0.7682, 0.0885]
print(a)

#
# Our dataset
#

# Let's say our dataset consists of 25 samples. Each sample is represented by a
# feature vector of shape (10,), meaning we have 10 features describing one
# sample. Create 25 random samples with 10 features in the range [-1, 1)
our_samples = rng.uniform(low=-1, high=1, size=(25, 10))

#
# torch.utils.data.Dataset
#

# The best way to utilize the convenient PyTorch data loading pipelines is by
# representing your dataset as class derived from the PyTorch dataset class
# torch.utils.data.Dataset.
# https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
from torch.utils.data import Dataset


class Simple1DRandomDataset(Dataset):
    
    def __init__(self, samples: np.ndarray):
        # super().__init__()  # Optional, since Dataset.__init__() is a no-op
        self.samples = samples
    
    # The __getitem__ method is required and should return a single sample based
    # on the provided index.
    def __getitem__(self, index):
        # Now we have to specify how to get the sample at "index":
        sample_features = self.samples[index]
        # It's a good idea to return the index/ID of the sample for debugging
        sample_id = index  # let's say that our "index" is the sample ID
        # And we have to return the sample (here: 2-tuple of data and ID)
        return sample_features, sample_id
    
    # The __len__ method is optional, but it is used by many samplers (see
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler for
    # more details). It should return the total number of samples.
    def __len__(self):
        return len(self.samples)


# Done! We have represented our dataset as PyTorch dataset!
our_dataset = Simple1DRandomDataset(our_samples)
print(f"our_dataset: {our_dataset}")
print(f"number of samples in our_dataset: {len(our_dataset)}")

#
# torch.utils.data.DataLoader
#

# Having our PyTorch compatible dataset, we can use the PyTorch DataLoader class
# to read the samples in minibatches.
# https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
from torch.utils.data import DataLoader

our_dataloader = DataLoader(
    our_dataset,  # We want to load our dataset
    shuffle=True,  # Shuffle the order of our samples
    batch_size=4,  # Stack 4 samples to a minibatch
    num_workers=0  # No background workers (see comment below)
)

# Note on "num_workers": If you want to run multiple workers, you will have to
# put the executing code within an 'if __name__ == "__main__":' guard. This
# if-condition is important to only execute the code in the main process, it
# will not work otherwise (in the background, the file is imported again, so the
# import must be done safely without executing the same multiprocessing code
# again). Further information:
# https://docs.python.org/3/library/multiprocessing.html
# https://pytorch.org/docs/stable/notes/multiprocessing.html
# For simplicity's sake, we will "num_workers=0" in this script.

# We can loop over our dataloader, and it will iterate over the dataset once,
# returning minibatches that contain our samples (each one as returned by the
# __getitem__ of the dataset):
for data in our_dataloader:
    mb_sample_features, mb_sample_ids = data
    print(f"Loaded samples {mb_sample_ids} with features:\n{mb_sample_features}\n")

# As we can see, the arrays have been stacked to minibatches and converted to
# PyTorch tensors, following the original datatype of the arrays:
print(f"ID: shape: {mb_sample_ids.shape}, dtype: {mb_sample_ids.dtype}")
print(f"features: shape: {mb_sample_features.shape}, dtype: {mb_sample_features.dtype}")

# This stacking is done by introducing a new first dimension. We will later see
# how to write custom stacking functions.

#
# torch.utils.data.Subset
#

# As we have already learned, it is important to split the training set from the
# test set (and optionally validation set). If you do not already have dedicated
# training, validation and test sets, and unless you need something very fancy,
# the PyTorch subset does exactly that.
from torch.utils.data import Subset

# Let's assign 1/5th of our samples to a test set, 1/5th to a validation set and
# the remaining 3/5th to a training set. We will use random splits.
n_samples = len(our_dataset)
shuffled_indices = rng.permutation(n_samples)
test_set_indices = shuffled_indices[:int(n_samples / 5)]
validation_set_indices = shuffled_indices[int(n_samples / 5):int(n_samples / 5) * 2]
training_set_indices = shuffled_indices[int(n_samples / 5) * 2:]

# Important: At this point, in a real ML project, you should save your subset
# indices to a file (e.g., .csv, .npz, or .pkl) for documentation and
# reproducibility!

# Create PyTorch subsets from our subset indices
test_set = Subset(our_dataset, indices=test_set_indices)
validation_set = Subset(our_dataset, indices=validation_set_indices)
training_set = Subset(our_dataset, indices=training_set_indices)

# Create dataloaders from each subset
test_loader = DataLoader(
    test_set,  # We want to load our test dataset
    shuffle=False,  # Do not shuffle test data
    batch_size=1  # 1 sample at a time
)
validation_loader = DataLoader(
    validation_set,  # We want to load our validation dataset
    shuffle=False,  # Do not shuffle validation data
    batch_size=4  # Stack 4 samples to a minibatch
)
training_loader = DataLoader(
    training_set,  # We want to load our training dataset
    shuffle=True,  # Shuffle the training data
    batch_size=4  # Stack 4 samples to a minibatch
)

# Let's try out our data loaders
print(f"test_set ({len(test_set)} samples)")
for mb_sample_features, mb_sample_ids in test_loader:
    print(f"Loaded samples {mb_sample_ids}")
print(f"validation_set ({len(validation_set)} samples)")
for mb_sample_features, mb_sample_ids in validation_loader:
    print(f"Loaded samples {mb_sample_ids}")
print(f"training_set ({len(training_set)} samples)")
for mb_sample_features, mb_sample_ids in training_loader:
    print(f"Loaded samples {mb_sample_ids}")


################################################################################
# PyTorch data handling - Dataset details
################################################################################

# In realistic scenarios, our dataset class might look a lot more complex. Not
# only can we add a custom __init__ method to it but also reading a sample can
# be much more sophisticated than just indexing. For example, let's create a
# dataset class that creates a simulated dataset of 10^15 random samples. Each
# sample is a sequence, described by a 2D array of input features of shape
# (sequence_length, n_features), with random values in the range [-1, 1). Each
# sample belongs to either a positive or negative class. In positive-class
# samples, we will implant the pattern [0, 1, 0, 1, 0, 1] in feature 0 at the
# beginning of the sequences. Since 10^15 is a large number, we do not want to
# hold that dataset in RAM. Instead, we create the random samples on-the-fly.


class RandomSeqDataset(Dataset):
    
    # Here, we define our __init__ method. In this case, we will take two
    # arguments, the sequence length "sequence_length" and the number of
    # features per sequence position "n_features".
    def __init__(self, sequence_length: int, n_features: int):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_samples = int(1e15)
        # We'll stay in float32, a typical datatype for GPU applications
        self.pattern = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
    
    # Here, we have to create a random sample and implant the pattern in
    # positive-class samples. Positive-class samples will have the label 1,
    # negative-class samples will have the label 0.
    def __getitem__(self, index):
        # While creating the samples randomly, we use the index as random seed
        # to get deterministic behavior (same ID/index = same sample)
        rnd_gen = np.random.default_rng(seed=index)
        
        # Create the random sequence of features. Use the method "random" and
        # scale to the range [-1, 1) manually to directly create a float32 data
        # array, since "uniform" returns a float64 array.
        sample_features = 2 * rnd_gen.random(size=(self.sequence_length, self.n_features), dtype=np.float32) - 1
        # Set every second sample to positive class
        sample_class_label = index % 2
        # Implant pattern in positive class
        if sample_class_label == 1:
            sample_features[:len(self.pattern), 0] = self.pattern
        
        # Here we could add pre-processing pipelines and/or normalization
        # ...
        
        # Let's say that our "index" is the sample ID
        sample_id = index
        # Return the sample, its label and its ID as a 3-tuple
        return sample_features, sample_class_label, sample_id
    
    def __len__(self):
        return self.n_samples


# Again, we can use a PyTorch dataloader to load our dataset:
training_set = RandomSeqDataset(sequence_length=16, n_features=9)
# Note: Shuffling would break with 10^15 sample indices
training_loader = DataLoader(training_set, shuffle=False, batch_size=4)

for i, data in enumerate(training_loader):
    mb_sample_features, mb_sample_labels, mb_sample_ids = data
    print(f"Loaded samples {mb_sample_ids}, labels {mb_sample_labels}, feature shape {mb_sample_features.shape}")
    # Maybe let's not do the full loop (unless you want to heat your CPUs)
    if i > 5:
        break


################################################################################
# PyTorch data handling - Minibatch stacking via "collate_fn"
################################################################################

# By default, the PyTorch DataLoader class will convert the sample arrays to
# tensors and stack them by introducing a new first dimension.
# In some cases, stacking arrays to minibatches will not work that easily. For
# example, if you want to stack samples with different shapes, such as sequences
# of variable length or images of variable width/height.
# In other cases, you may want to perform stacking to a minibatch in a very
# memory-efficient way, e.g., pre-allocating arrays for one-hot features.
# We can implement our own stacking function by passing the argument
# "collate_fn" to our dataloader, which should be our custom stacking function.
# https://pytorch.org/docs/stable/data.html
# https://pytorch.org/docs/stable/data.html#dataloader-collate-fn

#
# Simple example: No stacking of minibatch
#

# Function to be passed to torch.utils.data.DataLoader as "collate_fn". It will
# receive one argument: A list of the samples in the minibatch, as they were
# returned by the __getitem__ method of a PyTorch Dataset. Keep in mind that
# each sample is represented by a tuple, containing, e.g., the features, labels
# and IDs (see example above). In this example, instead of stacking the samples
# and converting them to tensors, the samples will be individually converted to
# tensors and packed into a list instead.
def no_stack_collate_fn(batch_as_list: list):
    # Number of entries per sample-tuple (e.g., 3 for features, labels, IDs)
    n_entries_per_sample = len(batch_as_list[0])
    # Go through all entries in all samples of the given list batch, convert
    # these entries to tensors and put them in lists
    list_batch = [[torch.tensor(sample[entry_i]) for sample in batch_as_list]
                  for entry_i in range(n_entries_per_sample)]
    # Return the minibatch
    return list_batch


# Let's try our "collate_fn" on our sequence dataset:
training_loader = DataLoader(training_set, shuffle=False, batch_size=4, collate_fn=no_stack_collate_fn)

for i, data in enumerate(training_loader):
    # Our minibatch is now a list of tensors instead of one stacked tensor!
    mb_sample_features, mb_sample_labels, mb_sample_ids = data
    print(f"Loaded samples {mb_sample_ids}, labels {mb_sample_labels}")
    if i > 5:
        break


#
# Example: Stacking and conversion of numpy arrays only
#

# Let's assume we want to return a sample tuple that consists of (features,
# labels, ID) but ID cannot be converted to a tensor (e.g., because it is a
# string).

# As example, we can inherit from our RandomSeqDataset class and overwrite the
# __getitem__ method such that the sample ID is a string:
class NewRandomSeqDataset(RandomSeqDataset):
    
    def __getitem__(self, index):
        # Call original __getitem__ from RandomSeqDataset class
        original_sample = super().__getitem__(index)
        sample_features, sample_class_label, sample_id = original_sample
        # Return the sample entries but convert sample_id to string
        return sample_features, sample_class_label, str(sample_id)


# Since strings cannot be converted to tensors or stacked, we have to stack and
# convert the sample entries only if they are stackable and convertible to
# tensors. We start by building the logic of converting-and-stacking-if-possible
# by implementing a function that will attempt to stack "something_to_stack" and
# convert it to a tensor. If not possible, "something_to_stack" will be returned
# as it was.
def stack_or_not(something_to_stack: list):
    try:
        # Convert to tensors (TypeError if it fails)
        tensor_list = [torch.tensor(s) for s in something_to_stack]
        # Try to stack tensors (RuntimeError if it fails)
        stacked_tensors = torch.stack(tensor_list, dim=0)
        return stacked_tensors
    except (TypeError, RuntimeError):
        return something_to_stack


# And now, we use it in our custom collate_fn, which will stack samples to a
# minibatch if possible
def stack_if_possible_collate_fn(batch_as_list: list):
    # Number of entries per sample-tuple (e.g., 3 for features, labels, IDs)
    n_entries_per_sample = len(batch_as_list[0])
    # Go through all entries in all samples and apply our "stack_or_not"
    list_batch = [stack_or_not([sample[entry_i] for sample in batch_as_list])
                  for entry_i in range(n_entries_per_sample)]
    # Return the minibatch (contains either stacked tensors, or lists of
    # elements that could not be converted to tensors)
    return list_batch


# We create a dataset instance of our new NewRandomSeqDataset
training_set = NewRandomSeqDataset(sequence_length=16, n_features=9)
# And load it via the PyTorch dataloader with our "stack_if_possible_collate_fn"
training_loader = DataLoader(training_set, shuffle=False, batch_size=4, collate_fn=stack_if_possible_collate_fn)

for i, data in enumerate(training_loader):
    # Our minibatch is now a stacked tensor where possible and a list otherwise!
    mb_sample_features, mb_sample_labels, mb_sample_ids = data
    print(f"Loaded samples {mb_sample_ids}, labels {mb_sample_labels}, feature shape {mb_sample_features.shape}")
    if i > 5:
        break


################################################################################
# PyTorch data handling - More examples
################################################################################

# See tasks/solutions files for:
# > Stacking multiple sequences of different lengths into a minibatch, using
#   padding.
# > Sending one-hot indices to the GPU and unpacking them into a minibatch.


################################################################################
# PyTorch data handling - Hints
################################################################################

# Setting global random seeds via, e.g., np.random.seed(0) will not make the
# order of samples or randomness in __getitem__ deterministic if the DataLoader
# is using multiple workers. Work-around for __getitem__ randomness: You can
# create a random generator object in __getitem__ which is using the sample
# index as random seed to always get the same random behavior for samples with
# the same index.
#
# Note: Pre-allocating memory is faster than first creating small arrays and
# then stacking them (unless PyTorch can optimize it using its magic).
#
# Using too many multiple worker processes will result in additional system
# overhead (managing the processes) and slow your program down! Always check
# the CPU utilization!
#
# It's good practice to save the indices of dataset splits in dedicated files.
