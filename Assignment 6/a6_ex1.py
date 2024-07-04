import numpy as np
import torch
import torchvision
import random
from torch.utils.data import Dataset

def augment_image(img_np: np.ndarray, index: int):
    transform_dict = {1:(torchvision.transforms.GaussianBlur(3),'GaussianBlur'),
                      2:(torchvision.transforms.RandomRotation(random.randint(0,360)),'RandomRotation'),
                      3:(torchvision.transforms.RandomVerticalFlip(),'RandomVerticalFlip'),
                      4:(torchvision.transforms.RandomHorizontalFlip(),'RandomHorizontalFlip'),
                      5:(torchvision.transforms.ColorJitter((0,1), (0,1), (0,1), (0,1)),'ColorJitter')}
    if index%7 == 0:
        return (torch.from_numpy(img_np), 'Original')
    elif index%7 in [1,2,3,4,5]:
        return (transform_dict[index%7][0](torch.from_numpy(img_np)), f'{transform_dict[index%7][1]}')
    elif index%7 == 6:
        unique_transforms = [1,2,3,4,5]
        used_transforms = []
        length = len(unique_transforms)
        for _ in range(3):
            i = random.randint(0,length-1)
            used_transforms.append(unique_transforms[i])
            unique_transforms.pop(i)
            length -= 1
        for i in used_transforms:
            if type(img_np) == np.ndarray:
                img_np = transform_dict[i][0](torch.from_numpy(img_np))
            else:
                img_np = transform_dict[i][0](img_np)
        return (img_np, 'Compose')
    
class TransformedImagesDataset(Dataset):
    def __init__(self, data_set: Dataset):
        self.dataset = data_set
    def __getitem__(self, index: int):
        img_info = list(self.dataset[index])
        transformed_img = augment_image(img_info[0], index)
        img_info[0] = transformed_img[0]
        img_info.insert(1, transformed_img[1])
        img_info.insert(2, index)
        return tuple(img_info)

    def __len__(self):
        return len(self.dataset)