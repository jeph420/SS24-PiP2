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

In this file, we will learn how to use PyTorch to train our NN layers using
gradient-based methods.
"""

################################################################################
# PyTorch - Creating a trainable parameter
################################################################################

# As shown in the previous unit, we can create trainable parameters in PyTorch
# using the torch.nn.Parameter class. It will return a tensor with trainable
# values and, by default, keep track of the gradients of the tensor.

import os
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)  # Set a known random seed for reproducibility

# Create tensor filled with random numbers from uniform distribution [0, 1)
param_values = torch.rand(size=(5, 1), dtype=torch.float32)
# Create trainable parameter from tensor values
trainable_param = nn.Parameter(data=param_values, requires_grad=True)


################################################################################
# PyTorch - Updating a trainable parameter
################################################################################

# We can use the automatic gradient computation via autograd to get the
# gradients of a computation w.r.t. our trainable parameter values.

# Assume we compute a value from our trainable parameter
output = trainable_param.sum() * 2

# We can get the computational graph for the gradient computation
print(f"output.grad_fn: {output.grad_fn}")

# We can compute the gradient of "output" w.r.t. a tensor with gradient
# information using autograd:
# (retain_graph=True if we want to compute the same gradients twice)
gradients = torch.autograd.grad(output, trainable_param, retain_graph=True)[0]
print(f"trainable_param gradients:\n{gradients}")

# Alternatively, we can call the convenience method "backward()", which will
# automatically compute the gradients of a scalar tensor w.r.t. all leaves of
# the computational graph, e.g., trainable tensors. The gradient values will be
# accumulated in the "grad" attribute of the graph leaves.
output.backward(retain_graph=True)
# The gradients that were computed are now accumulated in the nodes:
print(f"trainable_param.grad:\n{trainable_param.grad}")

# We have to reset the gradients explicitly, otherwise they will be accumulated:
output.backward()
print(f"trainable_param.grad (2nd time):\n{trainable_param.grad}")

# Resetting gradient
trainable_param.grad.zero_()
print(f"trainable_param.grad (reset):\n{trainable_param.grad}")


################################################################################
# PyTorch - Minimizing a loss (=optimizing our parameter values)
################################################################################

# Having the gradient values of some computation result w.r.t. the contributing
# trainable tensors allows us to use gradient descent methods to optimize the
# result. If this computation result is computed using a loss function, we will
# minimize the loss. "torch.optim" provides different optimization functions,
# such as stochastic gradient descent (SGD) or the Adam optimizer. We only have
# to supply a list of trainable parameters and specify optimizer-specific
# hyperparameters such as the learning rate.

# Assume we want our output value to be 1:
output = trainable_param.sum() * 2
target = torch.tensor(1, dtype=torch.float32)
loss = torch.abs(target - output)  # Absolute error as loss function
print("Initial:")
print(f"  trainable_param: {trainable_param}")
print(f"  output: {output}; target: {target}; loss: {loss}")

# Assume we want to use the SGD optimizer to optimize our trainable parameter
optimizer = torch.optim.SGD([trainable_param], lr=0.01)

# We can compute the gradients and then perform an update step:
loss.backward()
optimizer.step()
# We can reset the gradients easily using the optimizer:
optimizer.zero_grad()
print("After update:")
print(f"  trainable_param: {trainable_param}")
output = trainable_param.sum() * 2
loss = torch.abs(target - output)
print(f"  output: {output}; target: {target}; loss: {loss}")
# We decreased the loss by optimizing the trainable_param values using SGD!

# We can perform multiple update steps to further decrease our loss:
for update in range(5):
    # Compute the output
    output = trainable_param.sum() * 2
    # Compute the loss
    loss = torch.abs(target - output)
    # Compute the gradients
    loss.backward()
    # Perform the update
    optimizer.step()
    # Reset the accumulated gradients
    optimizer.zero_grad()
    print(f"Update {update + 1}/5:")
    print(f"  trainable_param: {trainable_param}")
    print(f"  output: {output}; target: {target}; loss: {loss}")

# We can add arbitrary computations to the tensor that should be optimized as
# long as PyTorch can compute a gradient. Let's say we want to have all values
# in trainable_param to be positive. We could simply add the absolute sum of its
# negative values to the loss. Note that this could lead to contradicting
# situations, e.g., finding the optimal solution is only possible with negative
# values, so the second, additional loss actually hinders the main loss, which
# is perfectly fine if you do not want a perfect solution in the first place
# (key terms: overfitting, regularization).
for update in range(50):
    # Compute the output
    output = trainable_param.sum() * 2
    # Compute the loss
    loss = torch.abs(target - output)
    # Add another loss term that minimizes negative values in trainable_param
    loss += trainable_param.clamp(max=0).sum().abs()
    # Compute the gradients
    loss.backward()
    # Preform the update
    optimizer.step()
    # Reset the accumulated gradients
    optimizer.zero_grad()
    if (update + 1) % 10 == 0:
        print(f"Update {update + 1}/50:")
        print(f"  trainable_param: {trainable_param}")
        print(f"  output: {output}; target: {target}; loss: {loss}")

# Note how we get closer to a loss of 0 while our main loss and additional loss
# term "compete", as the path we traverse using the SGD updates will depend on
# the values of the gradient computation. These gradient values will be bigger
# the higher the losses are, i.e., the update steps will be highest in direction
# of the highest decrease in loss. We could put more emphasis on the term that
# should keep the parameter values positive by using something like:
# loss += trainable_param.clamp(max=0).sum().abs() * 1e2
# Conversely, a scaling factor < 1 can be used for less emphasis.


################################################################################
# PyTorch - Optimizing parameters of PyTorch modules
################################################################################

# As shown in the previous unit, we can access the trainable parameter values of
# a PyTorch module using "parameters()". We can use this to train our model.

# Let's reuse the DSNN implementation from the previous unit:
class DSNN(nn.Module):
    
    def __init__(self, n_input_features: int, n_hidden_layers: int, n_hidden_units: int, n_output_features: int):
        """Fully-connected feed-forward neural network, consisting of
        ``n_hidden_layers`` linear layers, using selu activation function in the
        hidden layers.

        Parameters
        ----------
        n_input_features: int
            Number of features in input tensor
        n_hidden_layers: int
            Number of hidden layers
        n_hidden_units: int
            Number of units in each hidden layer
        n_output_features: int
            Number of features in output tensor
        """
        super().__init__()
        
        # We want to use n_hidden_layers linear layers
        hidden_layers = []
        for _ in range(n_hidden_layers):
            # Add linear layer module to list of modules
            layer = nn.Linear(in_features=n_input_features, out_features=n_hidden_units)
            torch.nn.init.normal_(layer.weight, 0, 1 / np.sqrt(layer.in_features))
            hidden_layers.append(layer)
            # Add selu activation module to list of modules
            hidden_layers.append(nn.SELU())
            n_input_features = n_hidden_units
        
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        # The output layer usually is separated to allow easy access to the
        # internal features (the model's data representation after the hidden
        # layers; see feature extraction example in 04_data_analysis.py)
        self.output_layer = nn.Linear(in_features=n_hidden_units, out_features=n_output_features)
        torch.nn.init.normal_(self.output_layer.weight, 0, 1 / np.sqrt(self.output_layer.in_features))
    
    def forward(self, x: torch.Tensor):
        """Apply deep SNN to ``x``.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(n_samples, n_input_features)`` or
            ``(n_input_features,)``

        Returns
        ----------
        torch.Tensor
            Output tensor of shape ``(n_samples, n_output_features)`` or
            ``(n_output_features,)``
        """
        # Apply hidden layers module
        hidden_features = self.hidden_layers(x)
        
        # Apply last layer (=output layer) without selu activation
        output = self.output_layer(hidden_features)
        return output


# Create an instance of our DSNN
dsnn = DSNN(n_input_features=5, n_hidden_layers=2, n_hidden_units=8, n_output_features=2)

# Create some input and target for our network
input_tensor = torch.arange(5, dtype=torch.float32)
target_tensor = torch.arange(2, dtype=torch.float32)

# "parameters()" will return all trainable parameters of the module, including
# the parameters of the submodules by default (recurse=True). Let's plug them
# into the SGD optimizer:
optimizer = torch.optim.SGD(dsnn.parameters(), lr=0.001)

# Optimize our dsnn model using SGD:
print("\nDSNN example:")
for update in range(50):
    # Compute the output
    output = dsnn(input_tensor)
    # Compute the loss
    # Important: The "backward" method will only work on scalars, so our loss
    # needs to be a scalar. In most cases, PyTorch's loss functions will take
    # care of this automatically. We compute the mean to create a scalar.
    loss = torch.abs(target_tensor - output).mean()
    # Compute the gradients
    loss.backward()
    # Preform the update
    optimizer.step()
    # Reset the accumulated gradients
    optimizer.zero_grad()
    if (update + 1) % 10 == 0:
        print(f"Update DSNN {update + 1}/50:")
        print(f"  output: {output}; target: {target_tensor}; loss: {loss}")

# Performing computations on different devices is straightforward
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dsnn = DSNN(n_input_features=5, n_hidden_layers=2, n_hidden_units=8, n_output_features=2).to(device=device)
input_tensor = torch.arange(5, dtype=torch.float32).to(device=device)
target_tensor = torch.arange(2, dtype=torch.float32).to(device=device)

optimizer = torch.optim.SGD(dsnn.parameters(), lr=0.001)

print("\nDSNN example:")
for update in range(50):
    output = dsnn(input_tensor)
    loss = torch.abs(target_tensor - output).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (update + 1) % 10 == 0:
        print(f"Update DSNN {update + 1}/50:")
        print(f"  output: {output}; target: {target_tensor}; loss: {loss}")


################################################################################
# PyTorch - Loss functions
################################################################################

# Pytorch offers different predefined optimizers and loss functions. Always make
# sure to check the documentation of the optimizer and loss function for their
# correct usage.
# https://pytorch.org/docs/stable/optim.html#algorithms
# https://pytorch.org/docs/stable/nn.functional.html#loss-functions
# https://pytorch.org/docs/stable/nn.html#loss-functions

#
# Mean squared error loss
# https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
#

# MSELoss can be used to, e.g., have a NN predict numerical values (="regression
# task").

# Let's assume a minibatch with 5 samples, 7 input features each
input_tensor = torch.rand(size=(5, 7), dtype=torch.float32)
# Let's assume each sample has 3 numerical target values:
target_tensor = torch.rand(size=(5, 3), dtype=torch.float32)
# Our network needs 3 output features to predict the 3 target values
dsnn = DSNN(n_input_features=7, n_hidden_layers=2, n_hidden_units=8, n_output_features=3)
# Define our MSE loss (reducing the loss of all samples to a scalar using the
# mean loss over the samples).
loss_function = torch.nn.MSELoss(reduction="mean")  # "mean" is the default
optimizer = torch.optim.SGD(dsnn.parameters(), lr=0.1)

print("\nMSE example:")
for update in range(50):
    output = dsnn(input_tensor)
    loss = loss_function(output, target_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (update + 1) % 10 == 0:
        print(f"Update {update + 1}/50:")
        print(f">> output:\n{output}")
        print(f">> target:\n{target_tensor}")
        print(f">> loss: {loss}")
        print("-" * 30)

#
# Binary classification task
# https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
#

# BCEWithLogitsLoss can be used to, e.g., have a NN predict mutually exclusive
# binary class labels using a sigmoid output activation function (="binary
# classification task").

# Let's assume a minibatch with 5 samples, 7 input features each
input_tensor = torch.rand(size=(5, 7), dtype=torch.float32)
# Let's assume each sample belongs to class 0 or 1:
target_tensor = torch.tensor([0, 0, 1, 1, 0], dtype=torch.float32).reshape((5, 1))
dsnn = DSNN(n_input_features=7, n_hidden_layers=2, n_hidden_units=8, n_output_features=1)
# This BCE implementation expects the values before applying the sigmoid
# activation function for numerical stability (see documentation):
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(dsnn.parameters(), lr=0.1)

print("\nBCE example:")
for update in range(50):
    output = dsnn(input_tensor)
    # To get the actual prediction of our network, we need to apply the sigmoid
    # function to its raw output. We do not need to do this when calculating the
    # loss, however, as BCEWithLogitsLoss already includes the sigmoid
    prediction = torch.sigmoid(output)
    loss = loss_function(output, target_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (update + 1) % 10 == 0:
        print(f"Update {update + 1}/50:")
        print(f">> output:\n{output}")
        print(f">> prediction:\n{prediction}")
        print(f">> target:\n{target_tensor}")
        print(f">> loss: {loss}")
        print("-" * 30)

#
# Multi-class classification task
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
#

# CrossEntropyLoss can be used to, e.g., have a NN predict multiple mutually
# exclusive class labels using softmax output activation function (="multi-class
# classification task").

# Let's assume a minibatch with 5 samples, 7 input features each
input_tensor = torch.rand(size=(5, 7), dtype=torch.float32)
# Let's assume each sample either belongs to class 0, 1 or 2:
target_tensor = torch.tensor([0, 2, 2, 1, 0], dtype=torch.long)
# We need 3 output features, one per class
dsnn = DSNN(n_input_features=7, n_hidden_layers=2, n_hidden_units=8, n_output_features=3)
# This CE implementation expects the values before applying the (log)softmax
# activation function for numerical stability (see documentation):
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(dsnn.parameters(), lr=0.1)

print("\nCE example:")
for update in range(50):
    output = dsnn(input_tensor)
    # Analogous comment as for BCE above, i.e., apply softmax to the raw model
    # output to get the predictions of class probabilities (but already included
    # in PyTorch's CrossEntropyLoss function via log-softmax)
    prediction = torch.softmax(output, dim=-1)
    loss = loss_function(output, target_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (update + 1) % 10 == 0:
        print(f"Update {update + 1}/50:")
        print(f">> output:\n{output}")
        print(f">> prediction:\n{prediction}")
        print(f">> target:\n{target_tensor}")
        print(f">> loss: {loss}")
        print("-" * 30)

#
# Multi-label classification task
#

# BCEWithLogitsLoss can be used to, e.g., have a NN predict multiple not
# mutually exclusive class labels using a sigmoid output activation function
# (="multi-label classification task").

# Let's assume a minibatch with 5 samples, 7 input features each
input_tensor = torch.rand(size=(5, 7), dtype=torch.float32)
# Let's assume each sample belongs to class 0, 1 or 2, or multiple classes:
target_tensor = torch.tensor([[0, 0, 1],
                              [1, 0, 1],
                              [0, 0, 0],
                              [1, 1, 1],
                              [1, 1, 0]], dtype=torch.float32)
# We need 3 output features, one per class
dsnn = DSNN(n_input_features=7, n_hidden_layers=2, n_hidden_units=8, n_output_features=3)
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(dsnn.parameters(), lr=0.1)

print("\nBCE (multi-label) example:")
for update in range(50):
    output = dsnn(input_tensor)
    prediction = torch.sigmoid(output)
    loss = loss_function(output, target_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (update + 1) % 10 == 0:
        print(f"Update {update + 1}/50:")
        print(f">> output:\n{output}")
        print(f">> prediction:\n{prediction}")
        print(f">> target:\n{target_tensor}")
        print(f">> loss: {loss}")
        print("-" * 30)


################################################################################
# Inspecting training - TensorBoard
################################################################################

# Inspecting your models during training is very important to understand their
# dynamics and find good models! TensorBoard offers a convenient way to monitor
# the training of your models. It supports histograms, line plots and other
# logging and visualization methods. It can be accessed via web-browser and
# stores results in a lossy manner. pip installation:
# pip install tensorboard (works without tensorflow)
# https://pytorch.org/docs/stable/tensorboard.html
# https://www.tensorflow.org/tensorboard/
import shutil  # For deleting the TensorBoard output directories
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # Progress bar (not necessary but useful)

#
# Using TensorBoard to track losses, weights and gradients during training
#

# Let's assume a minibatch with 5 samples, 7 input features each
input_tensor = torch.rand(size=(5, 7), dtype=torch.float32).to(device=device)
# Let's assume each sample has 3 numerical target values:
target_tensor = torch.rand(size=(5, 3), dtype=torch.float32).to(device=device)
dsnn = DSNN(n_input_features=7, n_hidden_layers=2, n_hidden_units=8, n_output_features=3).to(device=device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(dsnn.parameters(), lr=0.1)

# Define a TensorBoard summary writer that writes to the specified directory. We
# remove any existing directories, so we do not accumulate results when running
# this script multiple times. Of course, this is NOT something we would normally
# do, since we would delete all previous results!
log_dir = os.path.join("results", "experiment_00")
shutil.rmtree(log_dir, ignore_errors=True)
writer = SummaryWriter(log_dir)

print("\nTensorBoard example:")
for update in tqdm(range(3000), desc="training"):
    # Compute the output
    output = dsnn(input_tensor)
    # Compute the main loss
    main_loss = loss_function(output, target_tensor)
    # Add L2 regularization
    l2_term = torch.mean(torch.stack([(param ** 2).mean() for param in dsnn.parameters()]))
    # Compute final loss
    loss = main_loss + l2_term * 1e-2
    # Compute the gradients
    loss.backward()
    # Preform the update
    optimizer.step()
    
    if (update + 1) % 50 == 0:
        # Add losses as scalars to TensorBoard. Entries can be grouped using the
        # format "tag/value" for better separation in the web view afterward
        writer.add_scalar(tag="Training/main_loss", scalar_value=main_loss.cpu(), global_step=update)
        writer.add_scalar(tag="Training/l2_term", scalar_value=l2_term.cpu(), global_step=update)
        writer.add_scalar(tag="Training/loss", scalar_value=loss.cpu(), global_step=update)
        # Add parameters (weights) and gradients as arrays to TensorBoard
        for i, (name, param) in enumerate(dsnn.named_parameters()):
            writer.add_histogram(tag=f"Parameters/[{i}] {name}", values=param.cpu(), global_step=update)
            writer.add_histogram(tag=f"Gradients/[{i}] {name}", values=param.grad.cpu(), global_step=update)
    
    # Reset the accumulated gradients
    optimizer.zero_grad()

# Close (and flush) writer
writer.close()

# You can now start TensorBoard in a terminal using
# tensorboard --logdir SPECIFIED_OUTPUT_DIR --port=6060
# and then open http://localhost:6060/ in a web browser

# Of course, TensorBoard is not the only monitoring option. You can also store
# output yourself (CSV, JSON, raw results, images, etc.)


################################################################################
# Hints
################################################################################

#
# Weighting samples
#

# If the data set is unbalanced (e.g., 10% positive and 90% negative samples),
# it can help to either sample more positive samples per minibatch or,
# alternatively, increase the weight of the positive sample losses. Many loss
# functions allow for weighting classes, such as "weight" in
# torch.nn.CrossEntropyLoss().

#
# Learning rate and momentum
#

# The learning rate is a hyperparameter, i.e., you may have to optimize it to
# find a good learning rate. Learning rates should be in range [0, 1] and will
# depend on the magnitude of the loss values, the task and the optimizer
# algorithm. The default learning rates in PyTorch are good starting points. The
# same goes for the momentum, which also implicitly alters the learning rate and
# helps to overcome local minima and smooths gradients over samples.

#
# 16 bit computations
#

# If you use 16bit computations, you will probably have to increase the
# parameter for numerical stability if you use the Adam optimizer.

#
# Clipping gradients
#

# If training is unstable due to too strong outliers in the gradients, you can
# use gradient clipping to increase stability (at the cost of altering the
# gradients). Clipping value and method are hyperparameters.
clipping_value = 10
# You can either clip by norm:
torch.nn.utils.clip_grad_norm_(dsnn.parameters(), clipping_value)
# Or clip the values directly:
torch.nn.utils.clip_grad_value_(dsnn.parameters(), clipping_value)

#
# Regularization
#

# Optimizers in PyTorch already include a parameter "weight_decay", which is the
# scaling factor of the L2 weight penalty. Different optimizers might benefit
# from different versions of L2 weight penalty, so the PyTorch implementation
# should be preferred. You can/should still compute the L2 penalty explicitly
# for plotting in TensorBoard. Other common regularization methods are adding
# noise to inputs or features, L1 and L2 penalty or dropout. See
# https://pytorch.org/docs/stable/nn.html#dropout-layers for dropout layers.
# Warning: Prefer the dropout option of individual PyTorch modules if it exists
# (some layer classes require specialized dropout versions). Use
# torch.nn.AlphaDropout for networks using SELUs.

#
# Finding good hyperparameters
#

# Typically, hyperparameters influence each other and cannot be optimized
# independently of each other. If you have a lot of resources at your disposal,
# you can perform a larger grid search or random search over different
# hyperparameter combinations. However, in practice, you probably have to reduce
# the number of hyperparameters and the search space before you start a grid
# search or random search. To do this, you can manually check the training
# behavior and identify settings that would not work (e.g., learning rate far
# too low or high) and exclude those values/use them as boundaries. You may also
# be able to identify which magnitudes of differences in the value of a
# hyperparameter lead to performance differences and should be investigated.
# This is one of the things that make training NNs dependent on experience,
# since you will need to get a feeling for how certain hyperparameter values
# perform in different settings and combinations (in addition to the theoretical
# backgrounds of the hyperparameters).

#
# No-grad setting and model evaluation state
#

# When evaluating the model performance (e.g., on a validation set or test set),
# you typically do not want to compute gradients (better performance). This can
# be done, e.g., with the "torch.no_grad()" context manager. More information:
# https://pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation
# In some cases, your model might behave differently when training and when
# evaluating because certain (sub)modules  (e.g., Dropout) have a different
# behavior. If so, make sure to switch the model state to evaluation, e.g., via
# "model.eval()" before you run your evaluation code. Analogously, you can
# switch back to the training state via "model.train()". More information:
# https://pytorch.org/docs/stable/notes/autograd.html#evaluation-mode-nn-module-eval


################################################################################
# Saving trained models
################################################################################

# PyTorch offers convenient ways of saving and loading models:
# https://pytorch.org/tutorials/beginner/saving_loading_models.html

# Saving trainable parameters of a model
torch.save(dsnn, os.path.join("results", "trained_dsnn.pt"))

# Loading trainable parameters of a model (the module must already be defined)
same_dsnn = torch.load(os.path.join("results", "trained_dsnn.pt"))


################################################################################
# Putting it all together
################################################################################

# We will now create a data set with random samples that consist of 5 features
# with values between -1 and 1. The target for each sample is the mean of the
# squared feature values. We will train a DSNN to solve this task. We will
# combine the materials of the previous units.
from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):
    
    def __init__(self, n_features: int = 5):
        """Create random samples that consist of ``n_features`` features with
        values between -1 and 1. The target for each sample is the mean of the
        squared feature values.
        """
        self.n_features = int(n_features)
        self.n_samples = int(1e15)
    
    @staticmethod
    def _get_target(values):
        return (values ** 2).mean()
    
    def __getitem__(self, index):
        # While creating the samples randomly, we use the index as random seed
        # to get deterministic behavior, independent of who or when this method
        # is called (will return the same sample for the same "index")
        rng = np.random.default_rng(index)
        
        # Create the random sequence of features. Use the method "random" and
        # scale to the range [-1, 1) manually to directly create a float32 data
        # array, since "rng.uniform()" returns a float64 array.
        features = 2 * rng.random(size=self.n_features, dtype=np.float32) - 1
        target = self._get_target(features)
        
        # Let's say that our "index" is the sample ID
        sample_id = index
        # Return the sample, its label and its ID
        return features, target, sample_id
    
    def __len__(self):
        return self.n_samples


training_set = RandomDataset(n_features=5)
training_loader = DataLoader(
    training_set,
    shuffle=False,
    batch_size=4,
    num_workers=0
)
dsnn = DSNN(
    n_input_features=5,
    n_hidden_layers=4,
    n_hidden_units=32,
    n_output_features=1
).to(device=device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(dsnn.parameters(), lr=1e-3)

log_dir = os.path.join("results", "experiment_01")
shutil.rmtree(log_dir, ignore_errors=True)
writer = SummaryWriter(log_dir)

n_updates = 10000  # Number of updates to train for
update = 0  # Update counter
update_progress_bar = tqdm(total=n_updates, desc="updates")

while update < n_updates:
    for data in training_loader:
        mb_features, mb_targets, mb_ids = data
        mb_features = mb_features.to(device=device)
        mb_targets = mb_targets.to(device=device)
        
        # Compute the output
        output = dsnn(mb_features)[:, 0]
        # Compute the main loss
        main_loss = loss_function(output, mb_targets)
        # Add L2 regularization
        l2_term = torch.mean(torch.stack([(param ** 2).mean() for param in dsnn.parameters()]))
        # Compute final loss
        loss = main_loss + l2_term * 1e-2
        # Compute the gradients
        loss.backward()
        # Preform the update
        optimizer.step()
    
        if (update + 1) % 50 == 0:
            writer.add_scalar(tag="Training/main_loss",
                              scalar_value=main_loss.cpu(),
                              global_step=update)
            writer.add_scalar(tag="Training/l2_term",
                              scalar_value=l2_term.cpu(),
                              global_step=update)
            writer.add_scalar(tag="Training/loss",
                              scalar_value=loss.cpu(),
                              global_step=update)
            writer.add_scalars(main_tag="Training/output_target",
                               tag_scalar_dict=dict(output=output[0].cpu(), target=mb_targets[0].cpu()),
                               global_step=update)
            for i, (name, param) in enumerate(dsnn.named_parameters()):
                writer.add_histogram(tag=f"Parameters/[{i}] {name}",
                                     values=param.cpu(),
                                     global_step=update)
                writer.add_histogram(tag=f"Gradients/[{i}] {name}",
                                     values=param.grad.cpu(),
                                     global_step=update)
        
        # Reset the accumulated gradients
        optimizer.zero_grad()
        
        # Here we could also compute the scores on a validation set or store the
        # currently best model (best = lowest validation loss).
        
        update_progress_bar.update()
        # Increment update counter (one minibatch iteration done)
        update += 1
        
        # Leave the minibatch iteration loop if n_updates is reached
        if update >= n_updates:
            break

update_progress_bar.close()
writer.close()
torch.save(dsnn, os.path.join("results", "trained_dsnn.pt"))


################################################################################
# Organization of files
################################################################################

# How you organize your files is up to you. Often, the classes, functions and
# scripts are split into multiple files to make the code more modular, reusable
# and readable. In ML repos, you often find a file containing data set code, a
# file containing training code, a file containing the architectures and a main
# file that imports and combines the code from the other files. It can also be a
# good choice to put hyperparameter settings in configuration files, e.g., JSON
# files. The "example_project" (available for download) contains a small, full
# example ML project, putting together what we learned so far (and additional
# things that we will cover in later units).
