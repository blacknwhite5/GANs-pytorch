import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable

# GPU 사용여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 하이퍼 파라메터
latent_size = 100
batch_size = 128
num_epoech = 3


# 데이터 불러오기
### 이미지 전처리
transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

### LSUN 데이터 셋 다운로드
os.makedirs('../../data/lsun', exist_ok=True)
lsun = datasets.LSUN(root='../../data/lsun',
                     classes=['bedroom_train'],
                     transform=transform)

#### 데이터 로드
dataloader = DataLoader(dataset=lsun,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=2)


# 생성자 신경망
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.project_reshape = nn.Sequential(nn.Linear(latent_size, 1024*4*4))
        self.fractional_stride_conv = nn.Sequential(
                
                # (128, 1024, 4, 4) -> (128, 512, 8, 8)
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                
                # (128, 1024, 8, 8) -> (128, 256, 16, 16)
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                
                # (128, 256, 16, 16) -> (128, 128, 32, 32)
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                # (128, 128, 32, 32) -> (128, 3, 64, 64)
                nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh()
            )
         
    def forward(self, z):
        out = self.project_reshape(z)
        out = out.view(out.shape[0], 1024, 4, 4)
        img = self.fractional_stride_conv(out)
        return img

# 판별자 신경망
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.downsampling = nn.Sequential(
                
                # (128, 3, 64, 64) -> (128, 128, 32, 32)
                nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),

                # (128, 128, 32, 32) -> (128, 256, 16, 16)
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                
                # (128, 256, 16, 16) -> (128, 512, 8, 8)
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                
                # (128, 512, 8, 8) -> (128, 1024, 4, 4)
                nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2, inplace=True),
        )
        
        # flatten -> sigmoid
        self.discrimination = nn.Sequential(
                nn.Linear(1024*4*4, 1),
                nn.Sigmoid()
        )
         
    def forward(self, x):
        out = self.downsampling(x)
        out = out.view(out.shape[0], -1)
        result = self.discrimination(out)
        return result


# 가중치 초기화 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main():

    # D & G  네트워크 생성
    D = Discriminator().to(device)
    G = Generator().to(device)
    print(D)
    print(G)

    # weights 초기화
    D.apply(weights_init)
    G.apply(weights_init)

    # optimizer 정의
    D_optim = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    G_optim = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epoech):
        for i, (imgs, _) in enumerate(dataloader):
            real_img = imgs.to(device)

            # 노이즈 벡터 생성
            z = torch.randn(batch_size, latent_size).to(device)
            
            # 가짜 이미지 생성
            fake_img = G(z)
            D_fake = D(fake_img)
            D_real = D(real_img)
            
            # # # # #
            # Discriminator 학습
            # # # # #
            
            # D loss
            loss_D = -((D_real.log() + (1-D_fake).log()).mean()/2)

            # D 최적화
            D_optim.zero_grad()
            loss_D.backward()
            D_optim.step()
            

            # # # # #
            # Generator 학습
            # # # # #
            fake_img = G(z)
            D_fake = D(fake_img)

            # G loss
            loss_G = -(D(fake_img).log().mean()) 
            
            # G 최적화
            G_optim.zero_grad()
            loss_G.backward()
            G_optim.step()



            # 학습 진행사항 출
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, num_epoech, i, len(dataloader),
                                                                loss_D.item(), loss_G.item()))

            # 이미지 저장 (save per 500 iter)
            batches_done = epoch * len(dataloader) + i
            if batches_done % 500 == 0:
                save_image(fake_img.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)


if __name__ == '__main__':
    main()
