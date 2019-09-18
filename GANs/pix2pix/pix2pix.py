import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from model import UNet, Discriminator
from datasets import ImageDataset

# GPU 사용여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# images & pretrained 디렉터리 생성
os.makedirs('images', exist_ok=True)
os.makedirs('pretrained', exist_ok=True)
print('Directories created')

# 하이퍼 파라매터
num_epoch = 200
batch_size = 10
lambda_recon = 100

# 이미지 불러오기
### 이미지 전처리
transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

### facades 데이터셋
datasets = {'{}'.format(mode) : ImageDataset(root='../../data/facades/',
                                transforms=transforms,
                                mode=mode)
                                for mode in ['train', 'test']}

### 데이터 로드
dataloader = {'{}'.format(mode) : DataLoader(datasets[mode],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=1)
                                  for mode in ['train', 'test']}


if __name__ == '__main__':
    # 네트워크
    G = UNet().to(device)
    D = Discriminator().to(device)

    G_optim = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_optim = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epoch):
        for i, imgs in enumerate(dataloader['train']):
            A = imgs['A'].to(device)
            B = imgs['B'].to(device)
            
            # # # # #
            # Discriminator
            # # # # #
            G.eval()
            D.train()

            fake = G(B)
            D_fake = D(fake, B)
            D_real = D(A, B)

            loss_D = -((D_real.log() + (1-D_fake).log()).mean())
    
            D_optim.zero_grad()
            loss_D.backward()
            D_optim.step()

            # # # # #
            # Generator
            # # # # #
            G.train()
            D.eval()

            fake = G(B)
            D_fake = D(fake, B)

            loss_G = -(D_fake.mean().log()) + lambda_recon * torch.abs(A - fake).mean()

            G_optim.zero_grad()
            loss_G.backward()
            G_optim.step()

            # 학습 진행사항 출력
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, num_epoch, i*batch_size, len(datasets['train']),
                                                                loss_D.item(), loss_G.item()))


        # 이미지 저장
        save_image(torch.cat([A, B, fake], dim=3), 'images/{0:03d}.png'.format(epoch+1), nrow=2, normalize=True)
