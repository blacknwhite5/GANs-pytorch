import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def UpDownBlock(mode, in_channel, out_channel,
                        kernel_size, stride, padding, activation='relu', norm=True):
            if mode == 'down':
                layers = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
            elif mode == 'up':
                layers = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, 
                                             padding=padding, output_padding=padding)]


            if norm:
                layers.append(nn.InstanceNorm2d(out_channel))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            return layers

        def ResnetBlock(in_channel, out_channel,
                        kernel_size=4, stride=1, padding=1, activation='relu', norm=True):
            layers = []
            for i in range(2):
                layers.extend([nn.ReflectionPad2d(padding),
                               nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)])
                if norm:
                    layers.append(nn.InstanceNorm2d(out_channel))
                if activation == 'relu' and i != 1:
                    layers.append(nn.ReLU(inplace=True))
            return layers
        
        self.down_sample = nn.Sequential(nn.ReflectionPad2d(3),
                                         *UpDownBlock('down', 3, 64, 7, 1, 0),
                                         *UpDownBlock('down', 64, 128, 3, 2, 1),
                                         *UpDownBlock('down', 128, 256, 3, 2, 1))

        b1 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b2 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b3 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b4 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b5 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b6 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b7 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b8 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b9 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        self.resnet_blocks = nn.Sequential(b1, b2, b3, b4, b5, b6, b7, b8, b9)

        self.up_sample = nn.Sequential(*UpDownBlock('up', 256, 128, 3, 2, 1),
                                       *UpDownBlock('up', 128, 64, 3, 2, 1),
                                       nn.ReflectionPad2d(3),
                                       nn.Conv2d(64, 3, kernel_size=7, stride=1),
                                       nn.Tanh())

    def forward(self, x):
        x = self.down_sample(x)
        x = self.resnet_blocks(x)
        out = self.up_sample(x)
        return out



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def Block(in_channel, out_channel,
                  kernel_size=4, stride=2, padding=1, activation='lrelu', norm=True):
            layers = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_channel))
            if activation == 'lrelu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.block1 = nn.Sequential(*Block(3, 64, norm=False))
        self.block2 = nn.Sequential(*Block(64, 128))
        self.block3 = nn.Sequential(*Block(128, 256))
        self.block4 = nn.Sequential(*Block(256, 512, stride=1))
        self.block5 = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, padding=1))

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        out = self.block4(b3)
        return out

def weights_init(m, mean=0, std=0.02):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

