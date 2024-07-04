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

In this file, we will see how to apply some basic data augmentation methods
using PyTorch. We will then use torchvision "transforms" to chain multiple image
transformation methods in one pipeline.
"""

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

torch.random.manual_seed(0)  # Set a known random seed for reproducibility

#
# Here, we prepare an example input image as PyTorch tensor
#

# Converter: PIL -> tensor
pil_to_tensor = torchvision.transforms.functional.to_tensor
# Converter: tensor -> PIL
tensor_to_pil = torchvision.transforms.functional.to_pil_image

with Image.open("08_example_image.jpg") as image:  # Read a PIL image
    image = pil_to_tensor(image)  # Shape (C, H, W)
print(f"Image shape: {image.shape}")


################################################################################
# Additive input noise
################################################################################

# If we add noise to our input data, we have to consider possible changes in the
# input distribution. By using mean=0 for our noise, we will at least not change
# the mean of the feature values. The standard deviation will be a
# hyperparameter we will have to figure out through a hyperparameter search.
# Let's create a function that adds noise from a normal distribution to our
# tensor (not restricted to image data):

def add_normal_noise(input_tensor, mean: int = 0, std: float = 0.5):
    # Create the tensor containing the noise
    noise_tensor = torch.empty_like(input_tensor)
    noise_tensor.normal_(mean=mean, std=std)
    # Add noise to input tensor and return results
    return input_tensor + noise_tensor


# Let's apply our function and check the result for different noise stds
fig, axes = plt.subplots(1, 4)
axes[0].imshow(tensor_to_pil(image))
axes[0].set_xticks([])  # Remove xaxis ticks
axes[0].set_yticks([])  # Remove yaxis ticks
axes[0].set_title("Original image")
for i, std in enumerate([0.1, 0.5, 1.0]):
    image_noisy = add_normal_noise(image, std=std)
    # Make sure that image values are in valid range
    image_noisy = image_noisy.clamp(min=0, max=1)
    axes[i + 1].imshow(tensor_to_pil(image_noisy))
    axes[i + 1].set_xticks([])
    axes[i + 1].set_yticks([])
    axes[i + 1].set_title(f"Noise (std: {std})")
fig.tight_layout()
plt.close(fig)


################################################################################
# Input Dropout
################################################################################

# We can use the different versions of dropout modules in PyTorch. The simplest
# dropout version "torch.nn.Dropout" is dropping out random values. Note that
# this will drop out random values of a tensor (not restricted to image data),
# independent of the channels or spatial dimensions. The remaining values will
# be rescaled to keep the distribution of pixel values closer to the original
# distribution.
fig, axes = plt.subplots(1, 4)
axes[0].imshow(tensor_to_pil(image))
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title("Original image")
for i, p in enumerate([0.1, 0.5, 0.7]):
    simple_dropout = torch.nn.Dropout(p=p)
    image_dropout = simple_dropout(image)
    image_dropout = image_dropout.clamp(min=0, max=1)
    axes[i + 1].imshow(tensor_to_pil(image_dropout))
    axes[i + 1].set_xticks([])
    axes[i + 1].set_yticks([])
    axes[i + 1].set_title(f"Dropout (p: {p})")
fig.tight_layout()
plt.close(fig)


################################################################################
# Chaining image transformations with torchvision
################################################################################

# When dealing with image data, torchvision can be used to combine various image
# transformation methods ("transforms") using "torchvision.transforms.Compose".
# This class allows us to chain multiple image transformations, some of them
# working on PIL images, others working on PyTorch tensors (check doc).
#
# Workflow:
# 1. Create list of instances of transforms
# 2. Use list to create "torchvision.transforms.Compose" instance
# 3. Apply "torchvision.transforms.Compose" instance to image (e.g., in
#    "torch.utils.data.Dataset")
#
# Below, we will see examples on how to combine transforms.
# Documentation: https://pytorch.org/vision/stable/transforms.html

from torchvision import transforms

#
# Example: PIL -> tensor
#

# Create chain of transforms (only 1 transform in this case)
transform_chain = transforms.Compose([
    transforms.ToTensor()  # Transform a PIL or numpy array to a tensor
])
# Apply image transformations
with Image.open("08_example_image.jpg") as image:  # Read a PIL image
    transformed_image = transform_chain(image)  # Apply transforms chain
print(f"Transformed image dtype: {transformed_image.dtype}")
print(f"Transformed image [min, max]: [{transformed_image.min()}, {transformed_image.max()}]")
print(f"Transformed image shape: {transformed_image.shape}")

# Note: ToTensor() automatically converts the uint8 pixel values of a numpy
# array or PIL image with values in range [0, 255] to a tensor of torch.float,
# range [0.0, 1.0] and shape (C, H, W).

#
# Example: PIL -> RandomRotation -> tensor
#

# Create chain of transforms
transform_chain = transforms.Compose([
    transforms.RandomRotation(degrees=180),  # Rotate in range (-180, 180)
    transforms.ToTensor()  # Transform a PIL or numpy array to a tensor
])
# Apply image transformations
with Image.open("08_example_image.jpg") as image:  # Read a PIL image
    transformed_image = transform_chain(image)  # Apply transforms chain

fig, axes = plt.subplots(1, 2)
axes[0].imshow(image)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title("Original image")
axes[1].imshow(tensor_to_pil(transformed_image))
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_title("Random rotation")
fig.tight_layout()
plt.close(fig)


#
# Custom transformations
# We can define custom transformations using "torchvision.transforms.Lambda".
#

def wrap_add_normal_noise(mean: int = 0, std: float = 0.5):
    # transforms.Lambda expects function with single input argument, so create
    # a wrapper function which is returned here
    
    def noisy_image(input_tensor):
        input_tensor = add_normal_noise(input_tensor, mean, std)
        return torch.clamp(input_tensor, min=0, max=1)
    
    return noisy_image


noise_transform = transforms.Lambda(lambd=wrap_add_normal_noise(std=0.1))
# We can now use "noise_transform" in our transforms chain

#
# Example: PIL -> Resize -> ColorJitter -> RandomRotation -> RandomVerticalFlip
#          -> RandomHorizontalFlip -> tensor -> noise -> RandomErasing
#

# Create chain of transforms
transform_chain = transforms.Compose([
    transforms.Resize(size=100),  # Resize image to minimum edge = 100 pixels
    transforms.ColorJitter(),
    transforms.RandomRotation(degrees=180),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    noise_transform,
    transforms.RandomErasing()
])
# Apply image transformations
with Image.open("08_example_image.jpg") as image:  # Read a PIL image
    transformed_image = transform_chain(image)  # Apply transforms chain

fig, axes = plt.subplots(1, 2)
axes[0].imshow(image)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title("Original image")
axes[1].imshow(tensor_to_pil(transformed_image))
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_title("Multiple transforms")
fig.tight_layout()
plt.close(fig)

#
# Notes
#

# - Some transforms work on PIL images, others on tensors -> keep order and
#   position of transformation in mind.
# - You might need 2 transform chains, one for training and one for evaluation.


################################################################################
# torchvision: Functional transforms
################################################################################

# We can access the transformation functions of the torchvision transform
# classes directly (without randomness). To do this, we use the "functionals" of
# the transforms in "torchvision.transforms.functional".
# https://pytorch.org/vision/stable/transforms.html#functional-transforms
import torchvision.transforms.functional as TF

with Image.open("08_example_image.jpg") as image:  # Read a PIL image
    hflipped = TF.hflip(image)
    vflipped = TF.vflip(image)
    cropped_resized = TF.resized_crop(image, top=370, left=270, height=400, width=300, size=100)
    rotated = TF.rotate(image, angle=70)

fig, axes = plt.subplots(2, 3)
axes[0, 0].imshow(image)
axes[0, 0].set_xticks([])
axes[0, 0].set_yticks([])
axes[0, 0].set_title("Original image")
axes[1, 0].imshow(hflipped)
axes[1, 0].set_xticks([])
axes[1, 0].set_yticks([])
axes[1, 0].set_title("hflip")
axes[0, 1].imshow(vflipped)
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])
axes[0, 1].set_title("vflip")
axes[0, 2].remove()
axes[1, 1].imshow(cropped_resized)
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])
axes[1, 1].set_title("cropped and resized")
axes[1, 2].imshow(rotated)
axes[1, 2].set_xticks([])
axes[1, 2].set_yticks([])
axes[1, 2].set_title("rotated")
fig.tight_layout()
plt.close(fig)


################################################################################
# Combining PyTorch Dataset and transforms
################################################################################

# PyTorch Dataset and torchvision transforms can be freely combined. There are
# two recommended options, where both options perform the transformations in the
# __getitem__() method of the PyTorch Dataset and can thereby be performed via
# multiprocessing in combination with the PyTorch DataLoader.
from torch.utils.data import Dataset, Subset, DataLoader


#
# Option 1: Adding the transforms to the __getitem__() method of the Dataset.
# Ideal when we want to always apply transforms - for training and evaluation.
# E.g., converting images to grayscale or resizing them.
#

class SimpleRandomImageDataset(Dataset):
    
    def __init__(self, transforms: transforms.Compose = None):
        """Create random PIL images and optionally process them with torchvision
        transforms.
        
        Parameters
        ----------
        transforms : torchvision.transforms.Compose
            Optional: Chain of torchvision transforms to process image data
        """
        self.transforms = transforms
        self.np_image_shape = (5, 7, 3)  # H, W, C
        self.n_samples = 1000
    
    def _make_image(self, rnd_seed: int):
        rng = np.random.default_rng(rnd_seed)
        rnd_image = rng.integers(size=self.np_image_shape, low=0, high=256, dtype=np.uint8)
        rnd_image = Image.fromarray(rnd_image)
        return rnd_image
    
    def __getitem__(self, index: int):
        """Creates random PIL image and processes it with torchvision transforms.
        
        Parameters
        ----------
        index : int
            Sample index
        
        Returns
        ----------
        Image
            Random image after transforms
        """
        # Create some random image
        rnd_image = self._make_image(rnd_seed=index)
        # Apply transforms (if specified)
        if self.transforms is not None:
            rnd_image = self.transforms(rnd_image)
        # Return transformed image
        return rnd_image
    
    def __len__(self):
        """Return number of samples"""
        return self.n_samples


# Create the data_set and specify some transforms to apply:
sri_data_set = SimpleRandomImageDataset(
    transforms=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
)
# Split into training, validation and test set
sri_training_set = Subset(sri_data_set, indices=range(int(len(sri_data_set) * 3 / 5)))
sri_validation_set = Subset(sri_data_set, indices=range(int(len(sri_data_set) * 3 / 5), int(len(sri_data_set) * 4 / 5)))
sri_test_set = Subset(sri_data_set, indices=range(int(len(sri_data_set) * 4 / 5), len(sri_data_set)))

# Create DataLoaders for the 3 data_set splits
sri_training_loader = DataLoader(sri_training_set, batch_size=1, num_workers=0)
sri_validation_loader = DataLoader(sri_validation_set, batch_size=1, num_workers=0)
sri_test_loader = DataLoader(sri_test_set, batch_size=1, num_workers=0)

# We can now use the DataLoaders to load the samples. The transforms are applied
# to all 3 data set splits:
for training_mb, validation_mb, test_mb in zip(sri_training_loader, sri_validation_loader, sri_test_loader):
    # Plot the first minibatch of the data set splits as example, then exit loop
    print(f"Training mb: {training_mb.shape}\n{training_mb}")
    print(f"Validation mb: {validation_mb.shape}\n{validation_mb}")
    print(f"Test mb: {test_mb.shape}\n{test_mb}")
    break


#
# Option 2: Adding the transforms to the __getitem__() method of a new Dataset.
# Often, we want to apply different transforms to training and evaluation data.
# We can do this by creating a dedicated Dataset class that wraps a Dataset and
# only applies the transforms.
#

# First, we create a Dataset that returns samples without transforms:
class SimpleRandomImageDataset(Dataset):
    
    def __init__(self):
        self.np_image_shape = (5, 7, 3)  # H, W, C
        self.n_samples = 1000
    
    def _make_image(self, rnd_seed: int):
        rng = np.random.default_rng(rnd_seed)
        rnd_image = rng.integers(size=self.np_image_shape, low=0, high=256, dtype=np.uint8)
        rnd_image = Image.fromarray(rnd_image)
        return rnd_image
    
    def __getitem__(self, index: int):
        """Creates random PIL image.
        
        Parameters
        ----------
        index : int
            Sample index
        
        Returns
        ----------
        Image
            Random image after transforms
        """
        return self._make_image(rnd_seed=index)
    
    def __len__(self):
        """Return number of samples"""
        return self.n_samples


# Then, we create a Dataset that applies transforms to the output of the other
# Dataset:
class TransformsDataset(Dataset):
    
    def __init__(self, data_set: Dataset, transforms: transforms.Compose = None):
        """Apply transforms to the samples of the specified PyTorch Dataset.
        
        Parameters
        -------------
        data_set : torch.utils.data.Dataset
            Dataset to get samples from and apply the specified transforms to
        transforms : torchvision.transforms.Compose
            Optional: Chain of torchvision transforms to process image data
        """
        self.data_set = data_set
        self.transforms = transforms
    
    def __getitem__(self, index: int):
        """Get sample from specified PyTorch Dataset and apply transforms
        
        Parameters
        -------------
        index : int
            Sample index
        
        Returns
        -------------
        Image
            Random image after transforms
        """
        sample = self.data_set[index]
        # Apply transforms (if specified)
        if self.transforms is not None:
            sample = self.transforms(sample)
        # Return transformed sample
        return sample
    
    def __len__(self):
        return len(self.data_set)


# We can now create our Dataset and split it into our 3 sets:
sri_data_set = SimpleRandomImageDataset()
# Split into training, validation and test set
sri_training_set = Subset(sri_data_set, indices=range(int(len(sri_data_set) * 3 / 5)))
sri_validation_set = Subset(sri_data_set, indices=range(int(len(sri_data_set) * 3 / 5), int(len(sri_data_set) * 4 / 5)))
sri_test_set = Subset(sri_data_set, indices=range(int(len(sri_data_set) * 4 / 5), len(sri_data_set)))

# Now, we can apply transforms specific to the Dataset splits:
training_transforms = transforms.Compose([
    transforms.Resize(4),
    transforms.Grayscale(),
    transforms.RandomRotation(degrees=180),
    transforms.ToTensor()
])
evaluation_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])
# Wrap training set in TransformsDataset to apply training_transforms
sri_training_set = TransformsDataset(data_set=sri_training_set, transforms=training_transforms)
# Wrap evaluation sets in TransformsDataset to apply evaluation_transforms
sri_validation_set = TransformsDataset(data_set=sri_validation_set, transforms=evaluation_transforms)
sri_test_set = TransformsDataset(data_set=sri_test_set, transforms=evaluation_transforms)

# Create DataLoaders for the 3 data set splits
sri_training_loader = DataLoader(sri_training_set, batch_size=1, num_workers=0)
sri_validation_loader = DataLoader(sri_validation_set, batch_size=1, num_workers=0)
sri_test_loader = DataLoader(sri_test_set, batch_size=1, num_workers=0)

# We can now use the DataLoader to load the samples. The transforms are applied
# to all 3 data set splits:
for training_mb, validation_mb, test_mb in zip(sri_training_loader, sri_validation_loader, sri_test_loader):
    # Plot the first minibatch of the data_set splits as example, then exit loop
    print(f"Training mb: {training_mb.shape}\n{training_mb}")
    print(f"Validation mb: {validation_mb.shape}\n{validation_mb}")
    print(f"Test mb: {test_mb.shape}\n{test_mb}")
    break

# Notice how the image tensor in "training_mb" has a different shape since it
# went through different transforms than the tensors in "validation_mb" and
# "test_mb".
