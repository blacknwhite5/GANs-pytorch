from torch.utils.data import Dataset

import os
import glob
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
        return {'A':img_A, 'B':img_B}

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    from torchvision import transforms
    transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#    dataset = ImageDataset(root='../../data/facades/',
#                           transforms=transforms,
#                           mode='train')

    dataset = ImageDataset(root='../../data/facades/',
                           transforms=transforms,
                           mode='test')

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=8)

    import cv2
    import numpy as np
    for img in dataloader:
        A = img['A']
        B = img['B']
