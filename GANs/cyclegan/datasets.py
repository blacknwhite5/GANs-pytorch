from torch.utils.data import Dataset
from torchvision import transforms

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



if __name__ == '__main__':
   
    transform = [transforms.Resize(128),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))] 


    from torch.utils.data import DataLoader
    dataset = ImageDataset(root='../../data/monet2photo/', transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0
    )

    for i in dataloader:
        print(i['A'].shape)
