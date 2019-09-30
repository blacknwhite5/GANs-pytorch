from torch.utils.data import Dataset

import os
import glob
import random
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, transforms=None):
        Image.MAX_IMAGE_PIXELS = None
        self.transforms = transforms
        self.classes = {}
        for i, cls in enumerate(sorted(glob.glob(os.path.join(root, '*')))):
            self.classes[cls.split('/')[-1]] = i

        self.files = glob.glob(os.path.join(root, '*/*'))
        
    def __getitem__(self, idx):
        filename = self.files[idx % len(self.files)]
        cls = self.classes[filename.split('/')[-2]]

        img = Image.open(filename)
        # augmentation (5 crop - random choice)
        img = self.__fivecrop_randchoice(img)
        img = self.transforms(img)
        return img, cls

    def __len__(self):
        return len(self.files)

    def __fivecrop_randchoice(self, img):
        w, h = img.size
        dice = random.uniform(0,1)

        if dice >= 0.75:
            return img.crop((0, 0, w*.9,h*.9))   # top left
        elif dice < 0.75 and dice > 0.5:
            return img.crop((w*.1, 0, w, h*0.9))   # top right
        elif dice < 0.5 and dice  > 0.25:
            return img.crop((0, h*.1, w*.9, h))  # bottom left
        else:
            return img.crop((w*.1, h*.1, w, h)) # bottom right
