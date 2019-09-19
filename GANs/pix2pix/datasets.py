from torch.utils.data import Dataset

import os
import glob
import torch
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, transforms=None, mode='train'):
        self.transform = transforms
        self.files = sorted(glob.glob(os.path.join(root, '{}'.format(mode)) + '/*.*'))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        h, w = img.shape[1:]

        img_A = img[:,:,:w//2]
        img_B = img[:,:,w//2:]

        # flip
        if torch.randn(1) < 0.5:
            img_A = torch.flip(img_A, dims=(2,))
            img_B = torch.flip(img_B, dims=(2,))
            
        return {'A':img_A, 'B':img_B}

    def __len__(self):
        return len(self.files)
