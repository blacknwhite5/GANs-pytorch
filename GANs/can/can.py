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

    # 최적화
    D_optim = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    G_optim = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # loss 함수
    loss_BCE = nn.BCELoss()
    loss_CE = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        for i, (img, cls) in enumerate(dataloader):
            img = img.to(device)
            cls = cls.to(device)

            # 노이즈 생성
            z = torch.randn([batch_size,1,1,100]).to(device)

            # # # # #
            # Discriminator
            # # # # #
            fake = G(z)
            D_fake_cls, D_fake = D(fake)
            D_real_cls, D_real = D(img)

            loss_D_real = loss_BCE(D_real, torch.ones_like(D_real))
            loss_D_fake = loss_BCE(D_fake, torch.zeros_like(D_fake))
            loss_D_cls_real = loss_CE(D_real_cls, cls)

            loss_D = loss_D_real + loss_D_fake + loss_D_cls_real 
            
            D_optim.zero_grad()
            loss_D.backward(retain_graph=True)
            D_optim.step()


            # # # # #
            # Generator 
            # # # # #
            loss_G_fake = loss_BCE(D_fake, torch.ones_like(D_fake))
            loss_G_cls_fake = -((1/27)*torch.ones(batch_size, 27).to(device) \
                                    * nn.LogSoftmax(dim=1)(D_fake_cls)).sum(dim=1).mean()

            loss_G = loss_G_fake + loss_G_cls_fake

            G_optim.zero_grad()
            loss_G.backward()
            G_optim.step()


            # 학습 진행사항 출력
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, num_epoch, i*batch_size, len(wikiart),
                                                                loss_D.item(), loss_G.item()))


            # 이미지 저장 (save per epoch)
            batch_done = epoch * len(wikiart) + i
            if batch_done % 500 == 0:
                save_image(fake, 'images/{0:03d}.png'.format(batch_done), normalize=True)

if __name__ == '__main__':
    main()
