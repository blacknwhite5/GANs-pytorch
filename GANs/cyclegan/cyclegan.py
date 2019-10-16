import os
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from models import Generator, Discriminator, weights_init
from datasets import ImageDataset


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='../../data/monet2photo/', help='Directory of datasets')
parser.add_argument('--num_epoch', type=int, default=100, help='number of episode')
parser.add_argument('--batch_size', type=int, default=1, help='size of batch')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='first term of betas')
parser.add_argument('--b2', type=float, default=0.999, help='second term of betas')
parser.add_argument('--lambda_consist', type=int, default=10, help='lambda of consistency')
parser.add_argument('--lambda_id', type=int, default=5, help='lambda of identity')
parser.add_argument('--reuse', action='store_true', help='load pretrained model')
parser.add_argument('--save_path', type=str, default='pretrained/cyclegan.pth', help='A name of pretrained model file')
parser.add_argument('--num_workers', type=int, default=8, help='number of threads to use for data loading')
parser.add_argument('--sampling_interval', type=int, default=500, help='Interval of saving results from sampling')
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
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

### monet2photo 데이터셋
datasets = {'{}'.format(mode) : ImageDataset(root=args.datasets,
                                transforms=transforms,
                                mode=mode)
                                for mode in ['train', 'test']}

### 데이터 로드
dataloader = {'{}'.format(data) : DataLoader(datasets[data],
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)
                                  for data in ['train', 'test']}


def main():
    # D & G 네트워크 생성
    G = Generator().to(device)
    F = Generator().to(device)
    Dx = Discriminator().to(device)
    Dy = Discriminator().to(device)

    # 네트워크 초기화
    G.apply(weights_init)
    F.apply(weights_init)
    Dx.apply(weights_init)
    Dy.apply(weights_init)

    # pretrained model
    if args.reuse:
        assert os.path.isfile(args.save_path), '[!]Pretrained model not found'
        checkpoint = torch.load(args.save_path)
        G.load_state_dict(checkpoint['G'])
        F.load_state_dict(checkpoint['F'])
        Dx.load_state_dict(checkpoint['Dx'])
        Dy.load_state_dict(checkpoint['Dy'])
        print('[*]Pretrained model loaded')


    # optimizer 정의
    import itertools
    optimizer_D = optim.Adam(itertools.chain(Dx.parameters(), Dy.parameters()), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_G = optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=args.lr, betas=(args.b1, args.b2))

    # lr scheduler 정의
    optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda epoch : 1.0 - max(0, epoch - 100)/100)
    optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch : 1.0 - max(0, epoch - 100)/100)

    for epoch in range(args.num_epoch):
        for i, imgs in enumerate(dataloader['train']):
            x = imgs['X'].to(device)
            y = imgs['Y'].to(device)

            # # # # #
            # Discriminator 학습
            # # # # #
            Dx.train()
            Dy.train()
            G.eval()
            F.eval()

            # X
            x_fake = G(x)
            recov_x = F(x_fake)
            Dx_fake = Dy(x_fake)
            Dx_real = Dx(x)

            # Y
            y_fake = F(y)
            recov_y = G(y_fake)
            Dy_fake = Dx(y_fake)
            Dy_real = Dy(y)

#            # original loss
#            loss_GAN_D_xy = -((Dy_real.mean().log() + (1-Dx_fake).mean().log()))
#            loss_GAN_D_yx = -((Dx_real.mean().log() + (1-Dy_fake).mean().log()))

            # LSGAN(Least Square) loss
            loss_GAN_D_xy = ((Dy_real - 1)**2).mean() + (Dx_fake**2).mean()
            loss_GAN_D_yx = ((Dx_real - 1)**2).mean() + (Dy_fake**2).mean()

            # cycle-consistency loss
            loss_consistency_x = torch.abs(x - recov_x).mean()
            loss_consistency_y = torch.abs(y - recov_y).mean()

            loss_X2Y_D = loss_GAN_D_xy + args.lambda_consist * loss_consistency_x
            loss_Y2X_D = loss_GAN_D_yx + args.lambda_consist * loss_consistency_y
            loss_cycleGAN_D = loss_X2Y_D + loss_Y2X_D

            optimizer_D.zero_grad()
            loss_cycleGAN_D.backward()
            optimizer_D.step()
            
            # # # # #
            # Generator 학습
            # # # # #
            Dx.eval()
            Dy.eval()
            G.train()
            F.train()

            # X
            x_fake = G(x)
            recov_x = F(x_fake)
            Dx_fake = Dy(x_fake)

            # Y
            y_fake = F(y)
            recov_y = G(y_fake)
            Dy_fake = Dx(y_fake)

#            # original loss
#            loss_GAN_G_xy = -(Dx_fake.mean().log())
#            loss_GAN_G_yx = -(Dy_fake.mean().log())

            # LSGAN(Least Square) loss
            loss_GAN_G_xy = ((Dx_fake-1)**2).mean()
            loss_GAN_G_yx = ((Dy_fake-1)**2).mean()

            # cycle-consistency loss
            loss_consistency_x = torch.abs(x - recov_x).mean()
            loss_consistency_y = torch.abs(y - recov_y).mean()

            # identity mapping loss (painting -> photo)
            identity_loss_x = torch.abs(F(x) - x).mean()
            identity_loss_y = torch.abs(G(y) - y).mean()

            loss_X2Y_G = loss_GAN_G_xy + args.lambda_consist * loss_consistency_x + args.lambda_id * identity_loss_x 
            loss_Y2X_G = loss_GAN_G_yx + args.lambda_consist * loss_consistency_y + args.lambda_id * identity_loss_y
            loss_cycleGAN_G = loss_X2Y_G + loss_Y2X_G
            
            optimizer_G.zero_grad()
            loss_cycleGAN_G.backward()
            optimizer_G.step()

            # 학습 진행사항 출력
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, args.num_epoch, i, len(dataloader['train']),
                                                                loss_cycleGAN_D.item(), loss_cycleGAN_G.item()))

            # Sampling 이미지 저장
            batches_done = epoch * len(dataloader['train']) + i 
            if batches_done % args.sampling_interval == 0:
                val = next(iter(dataloader['test']))
                real_X = val['X'].to(device)
                real_Y = val['Y'].to(device)

                with torch.no_grad():
                    fake_X = G(real_X)
                    fake_Y = F(real_Y)
                    recov_X = F(fake_X)
                    recov_Y = G(fake_Y)

                x_img = torch.cat([real_X, fake_X, recov_X], dim=3)
                y_img = torch.cat([real_Y, fake_Y, recov_Y], dim=3)
                total_img = torch.cat([x_img, y_img], dim=2)
                save_image(total_img, 'images/%d.png' % batches_done, nrow=1, normalize=True)
        
        torch.save({
                    'G' : G.state_dict(),
                    'F' : F.state_dict(),
                    'Dx': Dx.state_dict(),
                    'Dy': Dy.state_dict()
                   }
                   , args.save_path)
        print('[*] model saved')

if __name__ == '__main__':
    main()
