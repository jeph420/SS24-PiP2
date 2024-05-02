from a3_ex1 import ImagesDataset
import torch
import numpy

def stacking(batch_as_list: list):
    # This function must work as 'collate_fn' in 'torch.utils.data.DataLoader'
    # Each 'image' must be stacked in the shape of (N, 1, H, W); N: batch_size, 1: brightness_channel, H: height, W: width
    # Also stack the 'class_id' in the shape (N, 1); N: batch_size, 1: class_id
    # Store both the 'class_id' and 'image_filepath' in separate lists
    # Return a 4-tuple: (stacked_images, stacked_class_ids, class_names, image_filepaths)
    images = [sample[0] for sample in batch_as_list]
    class_ids = [sample[1] for sample in batch_as_list]
    class_names = [sample[2] for sample in batch_as_list]
    image_filepaths = [sample[3] for sample in batch_as_list]

    torched_images = []
    torched_class_ids = []

    for i in images:
        torched_images.append(torch.from_numpy(i))

    for i in class_ids:
        torched_class_ids.append(torch.tensor([i]))

    stacked_images = torch.stack(torched_images, dim=0)
    stacked_class_ids = torch.stack(torched_class_ids, dim=0)

    return (stacked_images, stacked_class_ids, class_names, image_filepaths)
    