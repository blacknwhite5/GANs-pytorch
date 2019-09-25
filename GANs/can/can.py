import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.utils import save_image
import torchvision.transforms as transforms

from models import Discriminator, Generator
from datasets import ImageDataset 

# GPU 사용여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 하이퍼 파라매터
latent_size = 100
batch_size = 32
num_epoch = 100
lr = 0.0002


# 이미지 불러오기
### 이미지 전처리
transforms = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

### wikiart 데이터 셋
wikiart = ImageDataset(root='../../data/wikiart/', 
                       transforms=transforms) 

### 데이터 로드
dataloader = DataLoader(wikiart,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=8)


def main():
    # 네트워트
    G = Generator().to(device)
    D = Discriminator().to(device)

    D_optim = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    G_optim = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epoch):
        for i, (img, cls) in enumerate(dataloader):
            img = img.to(device)
            cls = cls.to(device)

            # 노이즈 생성
            z = torch.randn([batch_size,1,1,100]).to(device)

            # # # # #
            # Discriminator
            # # # # #
            x_hat = G(z)
            D_fake_cls, D_fake = D(x_hat)
            D_real_cls, D_real = D(img)

            loss_isArt = -((D_real.log() + (1-D_fake).log()).mean())

            D_real_cls = torch.argmax(D_real_cls, dim=1).float()
            D_fake_cls = torch.argmax(D_fake_cls, dim=1).float()
            loss_isExistStyle = -(D_real_cls.mean().log() - (1/27)*(D_fake_cls.mean().log()) \
                                                                + (1-1/27)*(1-D_fake_cls.mean().log()))

#            loss_isExistStyle = -(D_real_cls.log() - (1/27)*(D_fake_cls.log()) \
#                                                        + (1-1/27)*(1-D_fake_cls).log()).mean()
            print(loss_isExistStyle)

            # # # # #
            # Generator 
            # # # # #


if __name__ == '__main__':
    main()
