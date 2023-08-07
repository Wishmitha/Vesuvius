import cv2
import numpy as np
import os
import shutil

from PIL import Image


def resize_image(image_path, factor):
    # Open the image
    image = Image.open(image_path)

    # Get the original width and height
    width, height = image.size

    # Calculate the new width and height
    new_width = int(width * factor)
    new_height = int(height * factor)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    return resized_image


def resize_image_set(input_path, output_path):
    # Iterate through the folders and subfolders
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".png") or file.endswith(".tif"):
                # Get the relative path of the file
                relative_path = os.path.relpath(root, input_path)
                # Create the corresponding output folder structure
                output_folder = os.path.join(output_path, relative_path)
                os.makedirs(output_folder, exist_ok=True)

                # Read and process the image
                image_path = os.path.join(root, file)
                reszied_image = resize_image(image_path, factor=0.25)

                # Save the image in the output folder
                output_file = os.path.join(output_folder, file)
                reszied_image.save(output_file)


def create_image_crops(image, window_size, stride):
    crops = []
    height, width = image.shape[:2]

    # Calculate the number of crops in each dimension
    num_crops_height = (height - window_size) // stride + 1
    num_crops_width = (width - window_size) // stride + 1

    for i in range(num_crops_height):
        for j in range(num_crops_width):
            # Calculate the starting indices of the crop
            start_h = i * stride
            start_w = j * stride

            # Extract the crop from the image
            crop = image[start_h:start_h + window_size, start_w:start_w + window_size]
            crops.append(crop)

    return np.array(crops)


def crop_image_set(input_path, output_path, window_size, stride):
    # Iterate through folders and subfolders
    for root, dirs, files in os.walk(input_path):
        # Create corresponding folder structure in the output path
        relative_path = os.path.relpath(root, input_path)
        output_folder = os.path.join(output_path, relative_path)
        os.makedirs(output_folder, exist_ok=True)

        # Process image files in the current folder
        for file in files:
            # Check if the file is a PNG or TIFF image
            if file.endswith(".png") or file.endswith(".tif") or file.endswith(".jpg"):
                # Read the image
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                # Generate crops
                crops = create_image_crops(image, window_size, stride)

                # Create output folder for the current image
                image_output_folder = os.path.join(output_folder, os.path.splitext(file)[0] + "_crops")
                os.makedirs(image_output_folder, exist_ok=True)

                # Save the crops
                for i, crop in enumerate(crops):
                    crop_path = os.path.join(image_output_folder, f"crop_{i}.png")
                    cv2.imwrite(crop_path, crop)

                print(f"Processed {file}: {len(crops)} crops saved in {image_output_folder}")


def save_mask_indices(path, dataset):
    # Iterate through all image files in the folder
    if dataset == 'train':
        frag_names = ['1','2','3']
    else:
        frag_names = ['a','b']

    for frag in frag_names:

        for filename in os.listdir(os.path.join(os.path.join(path, frag),'mask_crops')):

            save_path = os.path.join(path, frag)

            if filename.lower().endswith(('.png')):
                image_path = os.path.join(os.path.join(os.path.join(path, frag),'mask_crops'), filename)

                # Load the image
                image = Image.open(image_path)

                # Convert the image to grayscale
                image = image.convert('L')

                # Convert the image to a NumPy array
                image_array = np.array(image)

                # Check if the median pixel value is 255
                if np.median(image_array) == 255:
                    # Save the file name in a text file
                    with open(os.path.join(save_path, 'masked_crops.txt'), 'a') as file:
                        file.write(filename + '\n')


if __name__ == '__main__':

    # input_path = "../../Datasets/vesuvius-challenge-ink-detection/test"
    # output_path = "../../Datasets/vesuvius-challenge-ink-detection/resized_test_250"
    # resize_image_set(input_path, output_path)

    input_path = "../../Datasets/alpub_v2/papyri_images/seg_mask_otsu"
    output_path = "../../Datasets/alpub_v2/papyri_images/seg_mask_otsu_256_200"
    crop_image_set(input_path, output_path, window_size=256, stride=200)

    #save_mask_indices('../../Datasets/vesuvius-challenge-ink-detection/cropped_test_256', dataset='test')

    pass