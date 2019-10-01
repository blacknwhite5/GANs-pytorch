import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.utils import save_image
import torchvision.transforms as transforms

from models import Discriminator, Generator, weights_init
from datasets import ImageDataset 

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='../../data/wikiart/', help='Directory of datsets')
parser.add_argument('--num_epoch', type=int, default=100, help='number of episode')
parser.add_argument('--batch_size', type=int, default=32, help='size of batch')
parser.add_argument('--latent_space', type=int, default=100, help='size of noise')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='first term of betas')
parser.add_argument('--b2', type=float, default=0.999, help='second term of betas')
parser.add_argument('--reuse', action='store_true', help='load pretrained model')
parser.add_argument('--save_path', type=str, default='pretrained/can.pth', help='the path of saving models')
parser.add_argument('--num_workers', type=int, default=8, help='number of threads to use for data loading')
parser.add_argument('--sampling_interval', type=int, default=500, help='Interval of saving results from sampling')
args = parser.parse_args()
print(args)

# GPU 사용여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# images & pretrained 디렉터리 생성
os.makedirs('images', exist_ok=True)
os.makedirs('pretrained', exist_ok=True)
print('[*]Directories created!')


# 이미지 불러오기
### 이미지 전처리
transforms = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])


### wikiart 데이터 셋
wikiart = ImageDataset(root=args.datasets, 
                       transforms=transforms) 

### 데이터 로드
dataloader = DataLoader(wikiart,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers)



def main():
    # 네트워트
    G = Generator().to(device)
    D = Discriminator().to(device)

    # weights 초기화
    G.apply(weights_init)
    D.apply(weights_init)
    print('[*]weights initialize done')

    # pretraind 모델 불러오기
    if args.reuse:
        assert os.path.isfile(args.save_path), '[!]Pretrained model not found'
        checkpoint = torch.load(args.save_path)
        G.load_state_dict(checkpoint['G'])
        D.load_state_dict(checkpoint['D'])
        print('[*]Pretrained models loaded')

    # 최적화
    D_optim = optim.Adam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    G_optim = optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # loss 함수
    loss_BCE = nn.BCELoss()
    loss_CE = nn.CrossEntropyLoss()

    for epoch in range(args.num_epoch):
        for i, (img, cls) in enumerate(dataloader):
            img = img.to(device)
            cls = cls.to(device)

            # 노이즈 생성
            z = torch.randn([args.batch_size,1,1,args.latent_space]).to(device)

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
            loss_G_cls_fake = -((1/27)*torch.ones(args.batch_size, 27).to(device) \
                                    * nn.LogSoftmax(dim=1)(D_fake_cls)).sum(dim=1).mean()

            loss_G = loss_G_fake + loss_G_cls_fake

            G_optim.zero_grad()
            loss_G.backward()
            G_optim.step()


            # 학습 진행사항 출력
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, args.num_epoch, i*args.batch_size, len(wikiart),
                                                                loss_D.item(), loss_G.item()))


            # 이미지 저장 (save per epoch)
            batch_done = epoch * len(wikiart) + i
            if batch_done % args.sampling_interval == 0:
                save_image(fake, 'images/{0:03d}.png'.format(batch_done), normalize=True)

            # 모델 저장
            torch.save({
                'G' : G.state_dict(),
                'D' : D.state_dict(),
                },
                args.save_path)

if __name__ == '__main__':
    main()
