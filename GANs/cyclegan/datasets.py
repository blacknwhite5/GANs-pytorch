from torch.utils.data import Dataset

import os
import glob
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, transforms=None, mode='train'):
        self.transform = transforms

        self.files_A = sorted(glob.glob(os.path.join(root, '{}/A'.format(mode)) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '{}/B'.format(mode)) + '/*.*'))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_B = Image.open(self.files_B[index % len(self.files_B)])

        image_A = self.transform(image_A)
        image_B = self.transform(image_B)
        return {'X':image_A, 'Y':image_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
