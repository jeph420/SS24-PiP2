import os
from PIL import Image
import re
from matplotlib.image import imread
import csv
import shutil

def pillowable_file(filename: str):
    try:
        with Image.open(filename) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False
    
def shape_verify(image_path):
    image = imread(image_path)
    if len(image.shape) == 3 and image.shape[0] >= 100 and image.shape[1] >= 100:
        return True

def check_RGB(image_path):
    with Image.open(image_path) as img:
        if img.mode == "RGB":
            return True

def file_finder_recursive(dir_path: str, search_level: int):

    for i in os.listdir(dir_path):

        if os.path.isfile(os.path.abspath(f"{dir_path}/{i}")): 
            file_list.append(os.path.abspath(f"{dir_path}/{i}"))
        
        elif os.path.isdir(os.path.abspath(f"{dir_path}/{i}")):
            file_finder_recursive(f"{dir_path}/{i}", search_level + 1)

def validate_images(input_dir: str, output_dir: str, log_file: str, formatter: str = "07d"):
    result_counter = 0

    if not os.path.isdir(os.path.abspath(input_dir)):
        raise ValueError(f"{input_dir} is not an existing directory")
    
    else:
        file_finder_recursive(input_dir, 1)

        with open(log_file, 'w') as log:
            for i in file_list:
                if re.search("jpg|jpeg", f"{input_dir}/{i}".lower()) == None: 
                    log.write(f"{i},1\n")
                    if os.path.getsize(f'{input_dir}/{i}') > 250000:
                        log.write(f"{i},2\n")
                        if not pillowable_file(f'{input_dir}/{i}'):
                            log.write(f"{i},3\n")
                            if not shape_verify(f'{input_dir}/{i}') and not check_RGB(f'{input_dir}/{i}'):
                                log.write(f"{i},4\n")
                                if os.path.abspath(f"{input_dir}/{i}") in file_list:
                                    log.write(f"{i},6\n")

        sorted_file_list = sorted(file_list)
        
        with open("labels.csv", 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow(["name","label"])
            
            for i in sorted_file_list:
                csvwriter.writerow([f"{sorted_file_list.index(i):{formatter}}.jpg",  
                                    f"{re.search('[a-z]+(?=.jpg|[0-9]+.jpg)', i).group()}"])
                shutil.copy(i, f"{os.path.abspath(output_dir)}/{sorted_file_list.index(i):{formatter}}.jpg")
                result_counter+=1
    return result_counter

file_list = []
sort_names = []
print(validate_images("_resized", "output_dir", "log_file.txt"))