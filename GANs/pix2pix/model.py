import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def DownBlock(in_channel, out_channel, norm=True):
            layers = [nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def UpBlock(in_channel, out_channel, activation='relu' ,norm=True, dropout=False):
            layers = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_channel))
            if dropout:
                layers.append(nn.Dropout(0.5))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())

            return layers


        self.d1 = nn.Sequential(*DownBlock(3, 64, norm=False))
        self.d2 = nn.Sequential(*DownBlock(64, 128))
        self.d3 = nn.Sequential(*DownBlock(128, 256))
        self.d4 = nn.Sequential(*DownBlock(256, 512))
        self.d5 = nn.Sequential(*DownBlock(512, 512))
        self.d6 = nn.Sequential(*DownBlock(512, 512))
        self.d7 = nn.Sequential(*DownBlock(512, 512))
        self.d8 = nn.Sequential(*DownBlock(512, 512, norm=False))

        self.u1 = nn.Sequential(*UpBlock(512, 512, dropout=True))
        self.u2 = nn.Sequential(*UpBlock(1024, 512, dropout=True))
        self.u3 = nn.Sequential(*UpBlock(1024, 512, dropout=True))
        self.u4 = nn.Sequential(*UpBlock(1024, 512))
        self.u5 = nn.Sequential(*UpBlock(1024, 256))
        self.u6 = nn.Sequential(*UpBlock(512, 128))
        self.u7 = nn.Sequential(*UpBlock(256, 64))
        self.u8 = nn.Sequential(*UpBlock(128, 3, activation='tanh', norm=False))

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1= torch.cat([self.u1(d8), d7], 1)
        u2= torch.cat([self.u2(u1), d6], 1)
        u3= torch.cat([self.u3(u2), d5], 1)
        u4= torch.cat([self.u4(u3), d4], 1)
        u5= torch.cat([self.u5(u4), d3], 1)
        u6= torch.cat([self.u6(u5), d2], 1)
        u7= torch.cat([self.u7(u6), d1], 1)
        out = self.u8(u7)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def Block(in_channel, out_channel, 
                  kernel_size=4, stride=2, padding=1, activation='lrelu', norm=True):
            layers = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
            if norm:
                layers.append(nn.BatchNorm2d(out_channel))
            if activation == 'lrelu':
                layers.append(nn.LeakyReLU(0.2))
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            return layers

        self.block1 = nn.Sequential(*Block(6, 64, norm=False))
        self.block2 = nn.Sequential(*Block(64, 128))
        self.block3 = nn.Sequential(*Block(128, 256))
        self.block4 = nn.Sequential(*Block(256, 512, stride=1))
        self.block5 = nn.Sequential(*Block(512, 1, stride=1, activation='sigmoid'))

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        out = self.block5(b4)
        return out
