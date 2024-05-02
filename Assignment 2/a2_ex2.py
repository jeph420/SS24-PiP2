import numpy as np

def crop_image(img: np.ndarray, crop_x, crop_y):
    y, x = img.shape
    start_x = (x-crop_x)//2
    start_y = (y-crop_y)//2    
    return img[start_y:start_y+crop_y,start_x:start_x+crop_x]

def pad_image(img: np.ndarray, pad_width, pad_height):
    rows, cols = img.shape
    height = (pad_height-rows)//2
    width = (pad_width-cols)//2
    return np.pad(img, ((height,), (width,)), mode='edge')

def image_subarea(array: np.ndarray, x: int, y: int, size: int):
    start_x, start_y = x, y
    end_x, end_y = x+size, y+size
    reshaped = np.reshape(array, (array.shape[1], array.shape[2]))
    subarea = reshaped[start_y:end_y, start_x:end_x]
    return np.reshape(subarea, (1, size, size))

def prepare_image(image: np.ndarray, width: int, height: int, x: int, y: int, size: int):

    image_height = image.shape[1]
    image_width = image.shape[2]

    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError("Expected a 3D image in the form: np.shape = (1, ..., ...)")
    elif width < 32 or height < 32 or size < 32:
        raise ValueError("Height, Width, or Size smaller than 32")
    else:
        if image_width < width or image_height < height:
            copied_image = np.reshape(image.copy(), (image_width,image_height))
            resized_image = pad_image(copied_image, width, height)
            resized_image = np.reshape(resized_image, 
                                       (1, resized_image.shape[0], resized_image.shape[1]))
            
        elif image_width > width or image_height > height:
            copied_image = np.reshape(image.copy(), (image_width,image_height))
            resized_image = crop_image(copied_image, width, height)
            resized_image = np.reshape(resized_image, 
                                       (1, resized_image.shape[0], resized_image.shape[1]))
            
        if x < 0 or (x + size) > resized_image.shape[2]:
            raise ValueError("Sub-area not bounded in Resized Image Width")
        elif y < 0 or (y + size) > resized_image.shape[1]:
            raise ValueError("Sub-area not bounded in Resized Image Height")
        
        subarea = image_subarea(resized_image, x, y, size)
    
        return tuple([resized_image, subarea])
    