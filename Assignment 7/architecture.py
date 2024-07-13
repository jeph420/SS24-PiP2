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

target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyCNN(1, [32, 64, 128], 3).to(target_device)

############ HELPER FUNCTIONS ############
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

class TransformedData(Dataset):
    def __init__(self, image_paths, transforms):
        self.image_paths = image_paths
        self.transform = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale if needed
        if self.transform:
            given_transforms = [i for i in range(len(self.transform))]
            length = len(given_transforms)
            chosen_transforms = []
            for _ in range(2):
                set_seed(42)
                i = random.randint(0,length-1)
                chosen_transforms.append(self.transform[i])
                given_transforms.pop(i)
                length -= 1
            image = transforms.Compose(chosen_transforms)(image)
        return image
##########################################

## Define Image Transformations
train_transformations = [
    transforms.GaussianBlur(3),
    transforms.RandomRotation(22.5),
    transforms.RandomHorizontalFlip(0.5),

]

#test_transformations = transforms.Compose([
#    transforms.PILToTensor(),
#    transforms.Normalize()
#])

## Load the datasets
original_data = ds.ImagesDataset(image_dir="training_data", dtype=int)

set_seed(42)
transformed_images = TransformedData(original_data.image_filepaths, train_transformations)
transformed_gray_images = [ds.to_grayscale(np.array(image, dtype=np.float32)) for image in transformed_images]
prepared_images = [ds.prepare_image(image, 100, 100, 0, 0, 32)[0] for image in transformed_gray_images]
normalized_images = [torch.tensor(i, dtype=torch.float32)/225.0 for i in prepared_images]
transformed_data = [(normalized_images[i], original_data[i][1], original_data[i][2], original_data[i][3]) for i in range(len(normalized_images))]

data = ConcatDataset([original_data, transformed_data])

## Create loss collectors
trainable_params = [i for i in model.parameters()]

## Create dataset splits for Training, Validation, and Testing
batch_size = 64
split_size = len(data)//100
train_split, test_split, val_split = 85*split_size, 15*split_size, 0*split_size

set_seed(42)
data_train, data_val, data_test = torch.utils.data.random_split(data, [0.80, 0.10, 0.10])

## Choose suitable Optimizer and No. Epochs
optimizer = torch.optim.Adam(trainable_params, lr=0.001, weight_decay=0.0001)
num_epochs = 20

## Set Seed for reproduceability and split the Dataset 'Randomly'
set_seed(42)
train_sample = torch.utils.data.SubsetRandomSampler(range(train_split))
val_sample = torch.utils.data.SubsetRandomSampler(range(train_split, train_split+val_split))
test_sample = torch.utils.data.SubsetRandomSampler(range(len(data)-test_split, len(data)))

train_loader = DataLoader(data_train, batch_size, train_sample)
val_loader = DataLoader(data_val, batch_size, val_sample)
test_loader = DataLoader(data_test, batch_size, test_sample)

#print(train_split, test_split, val_split)
#print(len(data_train), len(data_test), len(data_val))
#print(len(train_sample), len(test_sample), len(val_sample))
#print(len(train_loader), len(test_loader), len(val_loader))

# Train and Validate the CNN model 

train_loss = []
val_loss = []

for epoch in range(num_epochs):
        
        if len(val_loss) > 0:
            min_loss = min(val_loss)
        if epoch > 5 and min_loss not in val_loss[epoch-5:epoch]:
            break
    
        model.train()
        batch_loss = 0
        ## Loop over the DataLoader
    
        for data in train_loader:
        ## Split the train_loader into inputs and targets (the inputs will be fed to the network)
            inputs, targets, _, _= data
            inputs, targets = inputs.to(target_device), targets.to(target_device)
           
        ## Feed the inputs to the network, and retrieve outputs (don't forget to reset the gradients)
            optimizer.zero_grad()
            outputs = model(inputs)
        ## Compute Cross-Entropy Loss and Gradients of each batch.
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            batch_loss += loss
            loss.backward()
        ## Update network's weights according to the loss (above step).
            optimizer.step()
        train_loss.append(batch_loss/len(train_loader))
        
        ### Iterate over the entire eval_data - compute and store the loss of the data.
        model.eval()
        batch_loss = 0
        with torch.no_grad():
            for data in val_loader:
                ## Split the eval_loader into inputs and targets.
                inputs, targets, _, _= data
                inputs, targets = inputs.to(target_device), targets.to(target_device)
        
                ## Feed the inputs to the network, and retrieve outputs.
                outputs = model(inputs)
                ## Compute and accumilate the loss of the batch.
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                batch_loss += loss
                new_loss = batch_loss/len(val_loader)   
            val_loss.append(new_loss)
            ### Save the model with the lowest loss so far
            if len(val_loss) == 1:
                torch.save(model.state_dict(), "model.pth")
            elif min(val_loss) == new_loss:
                torch.save(model.state_dict(), "model.pth")

#data_test = torch.utils.data.random_split(data, [1.00])
#test_loader = DataLoader(data_test, batch_size)
model = MyCNN(1, [32, 64, 128], 3).to(target_device)  
model.load_state_dict(torch.load('model.pth'))
model.eval()
#print("Testing...")
#for i in test_loader:
#    print(i)

with torch.no_grad():
    TruePositives = 0
    Total = 0
    for data in test_loader:
        inputs, targets, _, _ = data
        inputs, targets = inputs.to(target_device), targets.to(target_device)
        outputs = model(inputs)
        prediction_batch = torch.argmax(nn.functional.softmax(outputs, dim=1), dim=1)
        for i in range(len(prediction_batch)):
            Total += 1
            if prediction_batch[i] == targets[i]:
                TruePositives += 1
    print(f"Accuracy: {100*(TruePositives/Total):.3f}%")

#torch.save(model.state_dict(), "model.pth")