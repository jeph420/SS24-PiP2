from glob import glob
from os import path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from a2_ex1 import to_grayscale
from a2_ex2 import prepare_image


class ImagesDataset(Dataset):

    def __init__(
            self,
            image_dir,
            width: int = 100,
            height: int = 100,
            dtype: Optional[type] = None
    ):
        self.image_filepaths = sorted(path.abspath(f) for f in glob(path.join(image_dir, "*.jpg")))
        class_filepath = [path.abspath(f) for f in glob(path.join(image_dir, "*.csv"))][0]
        self.filenames_classnames, self.classnames_to_ids = ImagesDataset.load_classnames(class_filepath)
        if width < 100 or height < 100:
            raise ValueError(f'width and height must be greater than or equal 100')
        self.width = width
        self.height = height
        self.dtype = dtype

    @staticmethod
    def load_classnames(class_filepath: str):
        filenames_classnames = np.genfromtxt(class_filepath, delimiter=';', skip_header=1, dtype=str)
        classnames = np.unique(filenames_classnames[:, 1])
        classnames.sort()
        classnames_to_ids = {}
        for index, classname in enumerate(classnames):
            classnames_to_ids[classname] = index
        return filenames_classnames, classnames_to_ids

    def __getitem__(self, index):
        with Image.open(self.image_filepaths[index]) as im:
            image = np.array(im, dtype=self.dtype)
        image = to_grayscale(image)  # Image shape is now (1, H, W)
        resized_image, _ = prepare_image(image, self.width, self.height, 0, 0, 32)
        classname = self.filenames_classnames[index][1]
        classid = self.classnames_to_ids[classname]
        return resized_image, classid, classname, self.image_filepaths[index]

    def __len__(self):
        return len(self.image_filepaths)


if __name__ == "__main__":
    dataset = ImagesDataset("./validated_images", 100, 100, int)
    for resized_image, classid, classname, _ in dataset:
        # print(f'image filepath: {image_filepath}')
        print(f'image shape: {resized_image.shape}, dtype: {resized_image.dtype}, '
              f'classid: {classid}, classname: {classname}\n')
