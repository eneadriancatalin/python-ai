import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class importData(Dataset):
    def __init__(self, csvFile, transforms=None):
        self.data = pd.read_csv(csvFile)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        label = torch.tensor(int(self.data.iloc[idx, 0]))
        image = Image.open(img_path)

        if self.transforms:
            image = self.transforms(image)

        return (image, label)

