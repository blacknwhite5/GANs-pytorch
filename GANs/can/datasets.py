from torch.utils.data import Dataset

import os
import glob
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.transforms = transforms
        self.classes = {}
        for i, cls in enumerate(sorted(glob.glob(os.path.join(root, '*')))):
            self.classes[cls.split('/')[-1]] = i

        self.files = glob.glob(os.path.join(root, '*/*'))
        
    def __getitem__(self, idx):
        filename = self.files[idx % len(self.files)]
        cls = self.classes[filename.split('/')[-2]]

        img = Image.open(filename)
        img = self.transforms(img)

        return img, cls

    def __len__(self):
        return len(self.files)
