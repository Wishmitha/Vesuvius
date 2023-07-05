import cv2
import numpy as np


def smoothen_image(image, kernel_size):
    # Apply median smoothing
    image = cv2.medianBlur(image, kernel_size)
    # Apply Gaussian smoothing
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return image


def apply_adaptive_thresholding(image):
    # Set the blockSize and C parameters as per your requirements
    adapt_thresholded_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                    blockSize=11, C=2)

    return cv2.bitwise_not(adapt_thresholded_image)


def apply_otsu_thresholding(image):
    # Find the threshold value using Otsu's method
    _, threshold_value = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply the thresholding
    return np.where(image > threshold_value, 255, 0).astype(np.uint8)


def apply_erosion_dialiation(image, kernel_size, no_iterations):
    # Define the structuring element for erosion and dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform dilation
    image = cv2.dilate(image, kernel, iterations=no_iterations)

    # Perform erosion
    image = cv2.erode(image, kernel, iterations=no_iterations)

    return image


def remove_disconnected_regions(image, min_region_size):
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    # Create a mask to filter out small regions
    mask = np.zeros_like(image, dtype=np.uint8)

    for label in range(1, num_labels):

        region_size = stats[label, cv2.CC_STAT_AREA]

        if region_size >= min_region_size:
            mask[labels == label] = 255

    # Apply the mask to the segmented image
    return cv2.bitwise_and(image, mask)
