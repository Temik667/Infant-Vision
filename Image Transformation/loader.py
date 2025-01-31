from torch.utils.data import Dataset
import cv2
from utils import get_image_paths, apply_color_perception, apply_blur


class InfantVisionDataset(Dataset):
    def __init__(self, img_dir, age_in_months, transform=True):
        self.img_dir = img_dir
        self.image_paths = get_image_paths(self.img_dir)
        self.transform = transform
        self.age_in_months = age_in_months

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = apply_color_perception(image, self.age_in_months)
            image = apply_blur(image, self.age_in_months)
        return image


