import PIL.Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset as ds
import random, cv2
import numpy as np
from PIL import Image

class MyCNN(nn.Module):
    def __init__(self, inputs, outputs, kernel, feature_nums):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, kernel)
        self.batch1 = nn.BatchNorm2d(feature_nums)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(inputs, outputs, kernel)
        self.batch2 = nn.BatchNorm2d(feature_nums)
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(inputs, outputs)

    def forward(self, input_images: torch.Tensor):
        x = self.conv1(input_images)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv2(input_images)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.linear(x)

        return x

model = MyCNN(100*100, 100*100, 1, 1)

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

def train_network(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer, device: torch.device = 'cpu') -> None:
    """
    Train specified network for one epoch on specified data loader.

    :param model: network to train
    :param data_loader: data loader to be trained on
    :param optimizer: optimizer used to train network
    :param device: device on which to train network
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for data, target in data_loader:
        data, target = data.float().to(device), target.long().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

data = ds.ImagesDataset("training_data")

batch_size = 32
test_size = len(data)//9
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

train_sample = torch.utils.data.SubsetRandomSampler(range(test_size, len(data)))
train_loader = DataLoader(data, batch_size, train_sample)
for i in train_loader:
    print(i[0],i[1])

test_sample = torch.utils.data.SubsetRandomSampler(range(test_size))
test_loader = DataLoader(data, batch_size, test_sample)

#train_network(model, train_loader, optimizer)