import numpy as np
import torch
from torch.utils.data import DataLoader

def stacking(batch_as_list: list):
    # Should be used as a 'collate_fn' of a torch.utils.data.Dataloader
    # Should work on the samples of TransformedImagesDataset
    # i.e. (trans_img, trans_name, index, class_id, class_name, img_path)

    images = []         # To be converted into TENSOR
    indices = []        # To be converted into TENSOR
    classids = []       # To be converted into TENSOR
    trans_names = []
    classnames = []
    img_paths = []

    for i in batch_as_list:
        images.append(i[0])
        indices.append([i[2]])
        classids.append([i[3]])

        trans_names.append(i[1])
        classnames.append(i[4])
        img_paths.append(i[5])

    # Each 'trans_img' must be stacked with resulting shape (N, 1, H, W)
    #       where N: batch size, 1: image channels, H: height, W: width
    #   Resulting type: torch.Tensor (Pytorch tensor)

    stacked_images = torch.from_numpy(np.stack(images))

    # Each 'index' or 'class_id' must also be stacked in shape (N, 1)
    #       where N: batch size (No. samples in a given batch)

    stacked_indices = torch.from_numpy(np.stack(indices))
    stacked_class_ids = torch.from_numpy(np.stack(classids))

    # Each 'trans_name', 'class_name', and 'img_path' should be returned in 3 separate lists
    #   No Conversion needed!

    return tuple([stacked_images, trans_names, 
                 stacked_indices, stacked_class_ids, 
                 classnames, img_paths])