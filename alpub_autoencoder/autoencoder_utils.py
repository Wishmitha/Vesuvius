import numpy as np
import matplotlib.pyplot as plt
import os

def list_subfolders(path):

    subfolder_list = []

    # Iterate over the folders in the specified directory
    for folder_name in os.listdir(path):
        folder_dir = os.path.join(folder_path, folder_name)
        # Check if the item is a directory
        if os.path.isdir(folder_dir):
            subfolder_list.append(folder_name)

    print(subfolder_list)

def read_alpub_data(path):

    pass

if __name__ == '__main__':
    ALPUB_PATH = '../../Datasets/alpub_v2'
    read_alpub_data(path=ALPUB_PATH)