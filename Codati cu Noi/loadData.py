import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
#read the image

class importData(Dataset):

    def __init__(self, csvFile):
        self.csvFile = pd.read_csv(csvFile)

    def __len__(self):
        return len(self.csvFile)

    def __getitem__(self, item):
        image = Image.open(self.csvFile.iloc[item, 1])
        label = torch.Tensor(self.csvFile.iloc[item, 0])
        image = transforms.ToTorch()(image)

        return (image, label)

