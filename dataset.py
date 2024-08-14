import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image[index])
        mask_path = os.path.join(self.mask_dir, self.image[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32)
        mask[mask == 255.0] = 1.0
        if self.transform is not None:
            transformed = self.transform(image = image,mask = mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask