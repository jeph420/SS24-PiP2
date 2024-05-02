from PIL import Image
import numpy as np
import os

def y_linear(R, G, B):
    return (0.2126*R) + (0.7152*G) + (0.0722*B)

def to_grayscale(pil_image: np.ndarray):
    with Image.open(pil_image) as img:
        
        img_copy = img.copy()
        image_shape = np.shape(img)
        if len(image_shape) == 2:
            return np.reshape(img_copy, (1, image_shape[0], image_shape[1]))
        elif len(image_shape) == 3 and image_shape[-1] != 3:
            raise ValueError("There should be exactly 3 color channels")
        elif len(image_shape) > 3:
            raise ValueError(f"Unsupported Image Shape: {image_shape}")
        else:
            rgb_list = list(img_copy.getdata())
            rgb_normal = []
            for pixel in rgb_list:
                normal_pixel = []
                for color in pixel:
                    normal_pixel.append(color/255)
                rgb_normal.append(normal_pixel)

            print(rgb_list[0], rgb_normal[0])

            for norm_pixel in rgb_normal:
                index = 0
                for C in norm_pixel:
                    if C <= 0.04045:
                        C = C/12.92
                        norm_pixel[index] = C
                        index += 1
                    else:
                        C = ((C+0.055)/1.055)**2.4
                        norm_pixel[index] = C
                        index += 1

            print(rgb_list[0], rgb_normal[0])
            print(y_linear(rgb_normal[0][0], rgb_normal[0][1], rgb_normal[0][2]))

            i = 0
            for norm_pixel in rgb_normal:
                linear_y = y_linear(norm_pixel[0], norm_pixel[1], norm_pixel[2])
                if linear_y <= 0.0031308:
                    y = 12.92*linear_y
                    rgb_normal[i] = (int(y*255),int(y*255),int(y*255))
                    i += 1
                else:
                    y = (1.055*(linear_y**(1/2.4)))-0.055
                    rgb_normal[i] = (int(y*255), int(y*255), int(y*255))
                    i += 1

            print(rgb_list[0], rgb_normal[0], type(rgb_normal[0]))
            print(img_copy.mode, img_copy.size)

            img_bw = Image.new(img_copy.mode, img_copy.size)
            img_bw.putdata(rgb_normal)
            img_bw.save('grauey.jpg')
            bright = img_bw.convert('L')
            bright.save('garay.jpg')
            bright = np.reshape(bright, (1, bright.size[0], bright.size[1]))
            print(np.shape(img_bw), np.shape(bright))

file_path = "/home/olegbushtyrkov/JKU_PiP2/Assignment 1/Python Photos/book.jpg"
to_grayscale(file_path)
