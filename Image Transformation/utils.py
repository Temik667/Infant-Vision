import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def get_image_paths(img_dir):
    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    return image_paths


def age_to_sigma(age_in_months):
    max_sigma = 4  # 20/600
    min_sigma = 0  # 20/20
    sigma = max_sigma - (age_in_months / 12) * (max_sigma - min_sigma)
    return sigma


def apply_blur(image, age_in_months):
    sigma = age_to_sigma(age_in_months)
    print(sigma)
    if sigma != 0.0:
        blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
        return blurred_image
    return image


def apply_color_perception(image, age_in_months):
    CT = chromatic_threshold(age_in_months)
    # Adjust saturation based on CT
    modified_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(float)
    modified_image[..., 1] *= CT  # Scale saturation channel
    modified_image = np.clip(modified_image, 0, 255).astype(np.uint8)
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_HSV2RGB)
    return modified_image


def chromatic_threshold(t):
    a = 1.0
    b = 0.1
    return a * (1 - np.exp(-b * t))


def display_images(images, titles):
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(num_images*3, 3))

    for i in range(num_images):
        axs[i].imshow(images[i])
        if titles is not None:
            axs[i].set_title(titles[i], fontsize=12)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()