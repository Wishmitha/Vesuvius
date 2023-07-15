import os
import numpy as np
import cv2


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
            print('Fragment:',frag,'Loaded:',file_name)
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
