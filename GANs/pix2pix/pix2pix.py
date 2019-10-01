import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from models import UNet, Discriminator, weight_init
from datasets import ImageDataset


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='../../data/facades/', help='Directory of datasets')
parser.add_argument('--num_epoch', type=int, default=200, help='number of episode')
parser.add_argument('--batch_size', type=int, default=1, help='size of batch')
parser.add_argument('--lambda_recon', type=int, default=100, help='lambda of reconstruction')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='first term of betas')
parser.add_argument('--b2', type=float, default=0.999, help='second term of betas')
parser.add_argument('--reuse', action='store_true', help='load pretrained model')
parser.add_argument('--save_path', type=str, default='pretrained/pix2pix.pth', help='A name of pretrained model file')
parser.add_argument('--num_workers', type=str, default=8, help='number of threads to use for data loading')
args = parser.parse_args()
print(args)

# GPU 사용여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# images & pretrained 디렉터리 생성
os.makedirs('images', exist_ok=True)
os.makedirs('pretrained', exist_ok=True)
print('[*]Directories created')


# 이미지 불러오기
### 이미지 전처리
transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

### facades 데이터셋
datasets = {'{}'.format(mode) : ImageDataset(root=args.datasets,
                                transforms=transforms,
                                mode=mode)
                                for mode in ['train', 'test']}

### 데이터 로드
dataloader = {'{}'.format(mode.split('/')[0]) : DataLoader(datasets[mode.split('/')[0]],
                                                batch_size=int(mode.split('/')[1]),
                                                shuffle=True,
                                                num_workers=args.num_workers)
                                                for mode in ['train/{}'.format(args.batch_size), 'test/10']}


def main():
    # 네트워크
    G = UNet().to(device)
    D = Discriminator().to(device)

    # 네트워크 초기화
    G.apply(weight_init)
    D.apply(weight_init)

    # pretrained 모델 불러오기
    if args.reuse:
        assert os.path.isfile(args.save_path), '[!]Pretrained model not found'
        checkpoint = torch.load(args.save_path)
        G.load_state_dict(checkpoint['G'])
        D.load_state_dict(checkpoint['D'])
        print('[*]Pretrained model loaded')

    # optimizer
    G_optim = optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    D_optim = optim.Adam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    for epoch in range(args.num_epoch):
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

            # original loss D
            loss_D = -((D_real.log() + (1-D_fake).log()).mean())

#            # LSGAN loss D
#            loss_D = ((D_real - 1)**2).mean() + (D_fake**2).mean()
    
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

            # original loss G
            loss_G = -(D_fake.mean().log()) + args.lambda_recon * torch.abs(A - fake).mean()

#            # LSGAN loss G
#            loss_G = ((D_fake-1)**2).mean() + args.lambda_recon * torch.abs(A - fake).mean()

            G_optim.zero_grad()
            loss_G.backward()
            G_optim.step()

            # 학습 진행사항 출력
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, args.num_epoch, i*args.batch_size, len(datasets['train']),
                                                                loss_D.item(), loss_G.item()))


        # 이미지 저장 (save per epoch)
        val = next(iter(dataloader['test']))
        real_A = val['A'].to(device)
        real_B = val['B'].to(device)

        with torch.no_grad():
            fake_A = G(real_B)
        save_image(torch.cat([real_A, real_B, fake_A], dim=3), 'images/{0:03d}.png'.format(epoch+1), nrow=2, normalize=True)

        # 모델 저장
        torch.save({
                'G' : G.state_dict(),
                'D' : D.state_dict(),
            },
            args.save_path)


if __name__ == '__main__':
    main()
