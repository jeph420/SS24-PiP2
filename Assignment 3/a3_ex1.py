import torch
import pandas as pd
import os
import PIL.Image
import numpy as np
from a2_ex1 import to_grayscale
from a2_ex2 import prepare_image

class ImagesDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir,
                  width: int = 100,
                  height: int = 100,
                  dtype = None):
                  
        self.image_dir = image_dir
        self.width = width
        self.height = height
        self.dtype = dtype
        self.image_paths = []
        self.classname_dict = {}
        self.classname_dump = []

        if width < 100 or height < 100:
            raise ValueError("Image size cannot be less than 100x100")
        else:
            classname_set = set()
            for file in os.listdir(self.image_dir):
                if '.csv' in file:
                    image_csv = pd.read_csv(f"{self.image_dir}/{file}", delimiter=';')
                    classid = 0
                    for classname in image_csv['label']:
                        self.classname_dump.append(classname)
                        classname_set.add(classname)
                        classid += 1
                else: 
                    self.image_paths.append(os.path.abspath(os.path.join(image_dir,file)))

            self.image_paths = sorted(self.image_paths)
            sorted_classname_list = sorted(classname_set)
            for classname in sorted_classname_list:
                self.classname_dict[classname]=sorted_classname_list.index(classname)

    def __getitem__(self, index):
        with PIL.Image.open(self.image_paths[index]) as img:
            if self.dtype != None:
                img_np = np.array(img, dtype=self.dtype)
            else:
                img_np = np.array(img)
            img_gray = to_grayscale(img_np)
            prepared_img = prepare_image(img_gray, self.width, self.height, x=0, y=0, size=32)
            return (prepared_img[1], self.classname_dict[self.classname_dump[index]], self.classname_dump[index], self.image_paths[index])

    def __len__(self):
        return len(self.image_paths)
    
