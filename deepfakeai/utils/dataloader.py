from torch.utils.data import Dataset
import os
from PIL import Image

class DeepfakeDataset(Dataset):
    """
    Custom Dataset Provider
    """
    def __init__(
        self, 
        root_dir, 
        transform=None
    ):
        self.samples = []
        self.transform = transform
        for label, folder in enumerate(["real", "fake"]):
            folder_path = os.path.join(root_dir, folder)
            for file in os.listdir(folder_path):
                self.samples.append((os.path.join(folder_path, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
