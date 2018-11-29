import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable


# GPU 사용여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 하이퍼 파라메터
latent_size = 64
batch_size = 100
num_epoch = 100


# 데이터 불러오기
### 이미지 전처리
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

### MNIST 데이터 셋 다운로드
os.makedirs('data/mnist', exist_ok=True)
mnist = datasets.MNIST(root='../data/mnist',
                    train=True, 
                    download=True, 
                    transform=transform)

### 데이터 로드
dataloader = DataLoader(dataset=mnist,
                    batch_size=batch_size,
                    shuffle=True)

# 신경망 구성
class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()

    def generator(self):
        return nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
            )

    def discriminator(self):
        return nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
            )


def main():

    # D & G 네트워크 생성
    gan = GAN()
    D = gan.discriminator().to(device)
    G = gan.generator().to(device)

    # optimizer 정의
    D_optim = optim.Adam(D.parameters(), lr=0.0002)
    G_optim = optim.Adam(G.parameters(), lr=0.0002)

    for epoch in range(num_epoch):
        for i, (img, _) in enumerate(dataloader):
            img = img.view(batch_size, -1).to(device)

            # # # # #
            # Discriminator 학습
            # # # # #
            z = torch.randn(batch_size, latent_size).to(device)
            fake_img = G(z)
            D_gene = D(fake_img)
            D_real = D(img)
            
            # D loss
            loss_D = -((D_real.log() + (1-D_gene).log()).mean())
            
            # D 최적화
            D_optim.zero_grad()
            G_optim.zero_grad()
            loss_D.backward()
            D_optim.step()


            # # # # #
            # Generator 학습 
            # # # # #
            fake_img = G(z)
            D_gene = D(fake_img)

            # G loss
            loss_G = -((D_gene).log().mean())
 
            # G 최적화
            D_optim.zero_grad()
            G_optim.zero_grad()
            loss_G.backward()
            G_optim.step()

            # 학습 진행사항 출력
            if (i+1) % 200 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                      .format(epoch, num_epoch, i+1, len(dataloader), loss_D.item(), loss_G.item(), 
                              D_real.mean().item(), D_gene.mean().item()))


        # 이미지 저장
        fake_img = fake_img.view(fake_img.size(0), 1, 28, 28)   # N * C * H * W
        save_image(fake_img, os.path.join('samples', 'fake_images-{}.png'.format(epoch+1)), nrow=10, normalize=True)

if __name__ == '__main__':
    main()