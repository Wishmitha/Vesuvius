import os
import numpy as np
import cv2
import re


def load_dataset(dataset_path, dataset):
    if dataset == 'train':
        path = os.path.join(dataset_path, 'cropped_train')
        frags = ['1', '2', '3']
    else:
        path = os.path.join(dataset_path, 'cropped_test')
        frags = ['a', 'b']

    # Read the file names from masked_crops.txt

    dataset = []

    for frag in frags:

        frag_path = os.path.join(path, frag)

        masked_crops_path = os.path.join(frag_path, 'masked_crops.txt')
        inklabels_folder = os.path.join(frag_path, 'inklabels_crops')
        surface_volume_folder = os.path.join(frag_path, 'surface_volume')

        with open(masked_crops_path, 'r') as file:
            file_names = [line.strip() for line in file]

        target_labels = []
        train_labels = []

        # Load the target label image and train label images
        for file_name in file_names:
            print('Fragment:', frag, 'Loaded:', file_name)
            # Construct the file paths for the target label and train labels
            target_label_path = os.path.join(inklabels_folder, file_name)
            train_label_paths = [os.path.join(surface_volume_folder, f'{i:02d}_crops', file_name) for i in range(65)]

            # Load the target label image
            target_label_image = cv2.imread(target_label_path)
            target_labels.append(target_label_image)

            # Load the train label images
            train_label_images = [cv2.imread(train_label_path) for train_label_path in train_label_paths]
            train_labels.append(train_label_images)

        # Convert the lists to numpy arrays for convenience
        target_labels = np.array(target_labels)
        train_labels = np.array(train_labels)

        dataset.append([train_labels, target_labels])

    return dataset


def get_crop_position(image_height, image_width, crop_number, window_size=70, stride=10):
    crop_row = crop_number // ((image_height - window_size) // stride + 1)
    crop_col = crop_number % ((image_width - window_size) // stride + 1)
    x = crop_col * stride
    y = crop_row * stride
    return x, y


def stitch_crops(image_height, image_width, image_crops, crop_positions):
    # Initialize output image with the desired size
    output_size = (image_height, image_width)
    output_image = np.zeros(output_size)

    # Iterate over the image crops and their positions
    for crop, position in zip(image_crops, crop_positions):
        crop = crop.cpu().numpy()[0][0]
        crop_height, crop_width = crop.shape[:2]
        x, y = get_crop_position(image_height, image_width, position)

        # Calculate the region occupied by the crop in the output image
        y_start, y_end = y, y + crop_height
        x_start, x_end = x, x + crop_width

        # Add the pixels of the crop to the corresponding region in the output image
        if crop.shape == output_image[y_start:y_end, x_start:x_end].shape:
            output_image[y_start:y_end, x_start:x_end] = output_image[y_start:y_end, x_start:x_end] + crop

    # Clip the pixel values to ensure they are within the desired range
    output_image = np.clip(output_image, 0, 255)

    # Convert the output image to the appropriate data type if needed
    output_image = output_image.astype(np.uint8)

    return  output_image


def get_crop_numer_list(file_name):
    # Read the text file
    with open(file_name, 'r') as file:
        content = file.read()

    # Extract the middle numbers using regular expression
    pattern = r'crop_(\d+)\.png'
    matches = re.findall(pattern, content)

    # Convert the matched strings to integers
    numbers = [int(match) for match in matches]

    return numbers
