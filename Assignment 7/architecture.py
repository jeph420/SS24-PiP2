import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset as ds
import random
import numpy as np
import tqdm

class MyCNN(nn.Module):
    def __init__(self, inputs, outputs, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs[0], kernel_size, padding=1)
        self.batch1 = nn.BatchNorm2d(outputs[0])
        self.soft1 = nn.Softmax(dim=1)
        self.conv2 = nn.Conv2d(outputs[0], outputs[1], kernel_size, padding=1)
        self.batch2 = nn.BatchNorm2d(outputs[1])
        self.soft2 = nn.Softmax(dim=1)
        self.conv3 = nn.Conv2d(outputs[1], outputs[2], kernel_size, padding=1)
        self.batch3 = nn.BatchNorm2d(outputs[2])
        self.soft3 = nn.Softmax(dim=1)
        self.linear = nn.Linear(outputs[2]*100*100, 20)

    def forward(self, input_images: torch.Tensor):
        x = self.conv1(input_images)
        x = self.batch1(x)
        x = self.soft1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.soft2(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.soft3(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

model = MyCNN(1, [32, 64, 128], 3)

############ HELPER FUNCTION ############
def set_seed(seed: int = 42) -> None:
    """
    Set seed for all underlying (pseudo) random number sources.

    :param seed: seed to be used
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#########################################

## Load the dataset
data = ds.ImagesDataset(image_dir="training_data", dtype=int)

## Create loss collectors
trainable_params = [i for i in model.parameters()]
training_loss_averages = []
eval_losses = []

## Create dataset splits for Training, Validation, and Testing
batch_size = 32
split_size = len(data)//9
train_split, test_split, val_split = 7*split_size, split_size, split_size

## Choose suitable Optimizer and No. Epochs
optimizer = torch.optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-4)
num_epochs = 10

## Set Seed for reproduceability and split the Dataset 'Randomly'
set_seed(42)
train_sample = torch.utils.data.SubsetRandomSampler(range(train_split))
test_sample = torch.utils.data.SubsetRandomSampler(range(train_split, train_split+test_split))
val_sample = torch.utils.data.SubsetRandomSampler(range(len(data)-val_split, len(data)))

train_loader = DataLoader(data, batch_size, train_sample)
test_loader = DataLoader(data, batch_size, test_sample)
val_loader = DataLoader(data, batch_size, val_sample)

## Train and Validate the CNN model 
for epoch in range(num_epochs):
        for _ in tqdm.tqdm(range(1), desc=f"Epoch {epoch}", ncols=100):
            model.train()
            batch_loss = 0
            ## Loop over the DataLoader
            for _ in tqdm.tqdm(range(num_epochs), desc=f"   Training...", ncols=100, leave=False):
                for data in train_loader:
                ## Split the train_loader into inputs and targets (the inputs will be fed to the network)
                    inputs, targets, _, _= data
                    
                ## Feed the inputs to the network, and retrieve outputs (don't forget to reset the gradients)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                ## Compute Cross-Entropy Loss and Gradients of each batch.
                    loss = torch.nn.functional.cross_entropy(outputs, targets)
                    loss.backward()
                ## Update network's weights according to the loss (above step).
                    optimizer.step()
                ## Collect batch-loss over the entire epoch - get average loss of the epoch.
                    batch_loss += loss
            training_loss_averages.append(batch_loss/len(train_loader))
            ### Iterate over the entire eval_data - compute and store the loss of the data.
            model.eval()
            batch_loss = 0
            for _ in tqdm.tqdm(range(num_epochs), desc=f"   Evaluation...", ncols=100, leave=False):
                for data in val_loader:
                    ## Split the eval_loader into inputs and targets.
                    inputs, targets, _, _= data
                    
                    ## Feed the inputs to the network, and retrieve outputs.
                    outputs = model(inputs)
                    ## Compute and accumilate the loss of the batch.
                    loss = torch.nn.functional.cross_entropy(outputs, targets)
                    batch_loss += loss
            eval_losses.append(batch_loss)
