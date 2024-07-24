import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import dataset as ds
import random
import numpy as np
import tqdm
import torchvision.transforms as transforms
from PIL import Image
from typing import OrderedDict

class MyCNN(nn.Module):
    def __init__(self, inputs, outputs, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs[0], kernel_size, padding=1)
        self.batch1 = nn.BatchNorm2d(outputs[0])
        self.soft1 = nn.Softmax(dim=1)
        self.conv2 = nn.Conv2d(outputs[0], outputs[1], kernel_size, padding=1)
        self.batch2 = nn.BatchNorm2d(outputs[1])
        self.soft2 = nn.Softmax(dim=1)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(outputs[1], outputs[2], kernel_size, padding=1)
        self.batch3 = nn.BatchNorm2d(outputs[2])
        self.soft3 = nn.Softmax(dim=1)
        self.drop = nn.Dropout(0.1)
        self.linear = nn.Linear(outputs[2]*100*25, 20)

    def forward(self, input_images: torch.Tensor):
        x = self.conv1(input_images)
        x = self.batch1(x)
        x = self.soft1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.soft2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.soft3(x)
        x = self.drop(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

model = MyCNN(1, [32, 64, 128], 3)