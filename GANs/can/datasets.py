from torch.utils.data import Dataset

import os
import glob
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.transforms = transforms
        self.classes = []
        for cls in glob.glob(os.path.join(root, '*')):
            self.classes.append(cls.split('/')[-1])

        self.files = glob.glob(os.path.join(root, '*/*'))
        
    def __getitem__(self, idx):
        filename = self.files[idx % len(self.files)]
        cls = filename.split('/')[-2]

        img = Image.open(filename)
        img = self.transforms(img)
        return img, cls

    def __len__(self):
        return len(self.files)



if __name__ == '__main__':
    from torchvision import transforms
    transforms = transforms.Compose([transforms.ToTensor()])

    dataset = ImageDataset('../../data/wikiart', transforms=transforms)

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    for i, (img, cls) in enumerate(loader):
        print(img.shape, cls)
