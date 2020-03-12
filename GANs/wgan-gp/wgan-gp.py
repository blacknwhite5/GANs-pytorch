from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import torch
import os

# GPU 사용여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# sample 이미지 저장경로
save_path = 'samples'
os.makedirs(save_path, exist_ok=True)

# 하이퍼 파라매터
latent_size = 64 
batch_size = 100 
num_epoch = 100
n_critic = 5
gp_value = 10

# 데이터 불러오기
### 이미지 전처리
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])])

### MNIST 데이터 셋 다운로드
os.makedirs('../../data', exist_ok=True)
mnist = datasets.MNIST(root='../../data',
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
            nn.Tanh())
    
    def discriminator(self):
        return nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            )


def main():

    # D & G 네트워크 생성
    gan = GAN()
    D = gan.discriminator().to(device)
    G = gan.generator().to(device)

    # optimizer 정의
    D_optim = optim.Adam(D.parameters(), lr=0.0001, betas=(0., 0.9))
    G_optim = optim.Adam(G.parameters(), lr=0.0001, betas=(0., 0.9))

    for epoch in range(num_epoch):
        for i, (img, _) in enumerate(dataloader):
            # # # # #
            # Discriminator 
            # # # # #
            for _ in range(n_critic):
                img = img.view(batch_size, -1).to(device)
                z = torch.randn(batch_size, latent_size).to(device)
                
                fake = G(z)
                D_fake = D(fake)
                D_real = D(img)

                eps = torch.rand((img.size(0), 1)).to(device)
                x_hat = eps*img + (1-eps)*fake
                d_hat = D(x_hat)

                ones = torch.ones(img.shape[0], 1, requires_grad=True).to(device)
                gradients = autograd.grad(outputs=d_hat,
                                          inputs=x_hat,
                                          grad_outputs=ones,
                                          create_graph=True,
                                          retain_graph=True,
                                          only_inputs=True)[0]
                gradients_penalty = ((gradients.norm(2, dim=1)-1)**2).mean()
                
                D_loss  = -torch.mean(D_real) + torch.mean(D_fake) + gp_value * gradients_penalty 
                D_optim.zero_grad()
                D_loss.backward()
                D_optim.step()

            # # # # #
            # Generator
            # # # # #
            fake = G(z)
            G_loss = -torch.mean(D(fake))
            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            if (i+1) % 200 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                    .format(epoch, num_epoch, i+1, len(dataloader), D_loss.item(), G_loss.item(),
                            D_real.mean().item(), D_fake.mean().item()))

        # 이미지 저장
        fake = fake.view(fake.size(0), 1, 28, 28)
        save_image(fake, os.path.join(save_path, 'fake_images-{}.png'.format(epoch+1)), nrow=10, normalize=True)



if __name__ == '__main__':
    main()
