import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from sklearn.preprocessing import OneHotEncoder


def list_subfolders(path):
    subfolder_list = []

    # Iterate over the folders in the specified directory
    for folder_name in os.listdir(path):
        folder_dir = os.path.join(path, folder_name)
        # Check if the item is a directory
        if os.path.isdir(folder_dir):
            subfolder_list.append(folder_name)

    return subfolder_list


def read_alpub_data(path):
    # List of character labels same as sub-folder names
    character_labels = list_subfolders(path)

    # Initialize empty lists to store images and labels
    images = []
    labels = []

    # Create an instance of the OneHotEncoder
    encoder = OneHotEncoder()
    # Fit the encoder on the character labels
    encoder.fit(np.array(character_labels).reshape(-1, 1))
    # Convert the character labels to one-hot encoding
    one_hot_labels = encoder.transform(np.array(character_labels).reshape(-1, 1)).toarray()
    one_hot_dict = {label: one_hot for label, one_hot in zip(character_labels, one_hot_labels)}

    # Loop through each character label
    for character in character_labels:
        character_folder = os.path.join(path, character)  # Path to the character folder

        print("Processing", character)

        # Loop through each image file in the character folder
        for image_file in os.listdir(character_folder):
            image_path = os.path.join(character_folder, image_file)  # Path to the image file

            image = plt.imread(image_path)
            label = one_hot_dict[character]

            # Append the preprocessed image and label to the lists
            images.append(image)
            labels.append(label)

    # Convert the image and label lists to numpy arrays
    out_labels = np.array(labels)
    images = np.array(images)

    return images, out_labels


def read_alpub_data_with_segmentation_masks(img_path, seg_path):
    # List of character labels same as sub-folder names
    character_labels = list_subfolders(img_path)

    # Initialize empty lists to store images and labels
    images = []
    labels = []
    masks = []

    # Create an instance of the OneHotEncoder
    encoder = OneHotEncoder()
    # Fit the encoder on the character labels
    encoder.fit(np.array(character_labels).reshape(-1, 1))
    # Convert the character labels to one-hot encoding
    one_hot_labels = encoder.transform(np.array(character_labels).reshape(-1, 1)).toarray()
    one_hot_dict = {label: one_hot for label, one_hot in zip(character_labels, one_hot_labels)}

    # Loop through each character label
    for character in character_labels:
        character_folder = os.path.join(img_path, character)  # Path to the character folder

        print("Processing Image", character)

        # Loop through each image file in the character folder
        for image_file in os.listdir(character_folder):
            image_path = os.path.join(character_folder, image_file)  # Path to the image file

            image = plt.imread(image_path)
            label = one_hot_dict[character]

            # Append the preprocessed image and label to the lists
            images.append(image)
            labels.append(label)

    # Loop through segmentation masks folder

    seg_files = sorted(os.listdir(seg_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    for seg_file in seg_files:
        seg_image_path = os.path.join(seg_path, seg_file)

        print("Processing Masks:", seg_image_path)

        image = plt.imread(seg_image_path)

        masks.append(image)

    # Convert the image and label lists to numpy arrays
    out_labels = np.array(labels)
    images = np.array(images)
    masks = np.array(masks)

    return images, out_labels, masks


def read_payri_images(path):
    image_list = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                image_path = os.path.join(root, file)
                print("Reading",image_path)
                try:
                    img = plt.imread(image_path)
                    img_array = np.array(img)
                    image_list.append(img_array)
                except Exception as e:
                    print(f"Error reading image '{image_path}': {e}")

    return np.array(image_list)


def normalize_masks(masks):
    masks_list = []

    for mask in masks:
        mask = np.round(mask / 255)
        masks_list.append(mask)

    return np.array(masks_list)

# if __name__ == '__main__':
#     ALPUB_PATH = '../../Datasets/alpub_v2/images'
#     X,Y = read_alpub_data(path=ALPUB_PATH)
