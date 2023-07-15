import cv2
import numpy as np

from PIL import Image
import os


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
                reszied_image = resize_image(image_path, factor=0.07)

                # Save the image in the output folder
                output_file = os.path.join(output_folder, file)
                reszied_image.save(output_file)


# Example usage
if __name__ == '__main__':
    input_path = "../../Datasets/vesuvius-challenge-ink-detection/test"
    output_path = "../../Datasets/vesuvius-challenge-ink-detection/resized_test"
    resize_image_set(input_path, output_path)
