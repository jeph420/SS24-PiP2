import numpy as np

def y_linear(R, G, B):
    return (0.2126*R) + (0.7152*G) + (0.0722*B)

def to_grayscale(pil_image: np.ndarray):

    if pil_image.ndim == 2:
        return np.expand_dims(pil_image.copy(), axis=0)
    
    elif pil_image.ndim == 3 and pil_image.shape[2] == 3:
        
        normalized_image = pil_image/255.0
        
        c_linear = np.where(normalized_image <= 0.04045,
                            normalized_image/12.92,
                            ((normalized_image+0.055)/1.055)**2.4)
        
        linear_y = y_linear(c_linear[:,:,0], c_linear[:,:,1], c_linear[:,:,2])
        
        y = np.where(linear_y <= 0.0031308,
                     12.92*linear_y,
                     (1.055*(linear_y**(1/2.4)))-0.055)
        
        grayscale_image = np.round(y*255).astype(pil_image.dtype)
        
        return np.expand_dims(grayscale_image, axis=0)
    else:
        raise ValueError("Invalid image shape. Expected a 2D or 3D image with 2-3 channels.")
