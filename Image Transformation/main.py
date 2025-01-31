from utils import display_images, apply_blur, apply_color_perception, get_image_paths
from loader import InfantVisionDataset
from torch.utils.data import Subset, DataLoader
import time
import cv2
import matplotlib.pyplot as plt

image_dir = 'volcano'


def show_loader_images():
    dataset = InfantVisionDataset(image_dir, age_in_months=2, transform=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for images in dataloader:
        print(images.shape)
        display_images(images)


def compare_load_times():
    dataset = InfantVisionDataset(image_dir, age_in_months=2, transform=False)
    subset = Subset(dataset, list(range(100)))
    dataloader = DataLoader(subset, batch_size=8, shuffle=True, num_workers=0)
    start_time = time.time()
    for _ in dataloader:
        pass
    end_time = time.time()
    exec_times = [(end_time - start_time) * 1000]
    print(f"Time to load data without transformations: {exec_times[0]:.2f} ms")

    dataset = InfantVisionDataset(image_dir, age_in_months=2, transform=True)
    subset = Subset(dataset, list(range(100)))
    dataloader = DataLoader(subset, batch_size=8, shuffle=True, num_workers=0)
    start_time = time.time()
    for _ in dataloader:
        pass
    end_time = time.time()
    exec_times.append((end_time - start_time) * 1000)
    print(f"Time to load data with transformations: {exec_times[1]:.2f} ms")

    plt.figure(figsize=(5, 5))
    bars = plt.bar(["Without Transform", "With Transform"],
                   exec_times, color=['blue', 'green'], alpha=0.7, width=0.4)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval - 1, f'{yval:.2f} ms', ha='center', va='bottom', fontsize=12)

    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Execution Time (ms)", fontsize=12)
    plt.title("Execution Time Comparison", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()


def simulate_vision_by_time(property_name="acuity"):
    months = [1, 6, 12]
    image_paths = get_image_paths(image_dir)[:10]
    for image_path in image_paths:
        transformed_images = []
        for month in months:
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            if property_name == "color":
                modified_image = apply_color_perception(original_image, month)
            else:
                modified_image = apply_blur(original_image, month)
            transformed_images.append(modified_image)
        display_images(transformed_images, [f'Month: {month}' for month in months])


show_loader_images()
compare_load_times()
simulate_vision_by_time("color")
simulate_vision_by_time("color")
