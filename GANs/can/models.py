import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        def block(in_channel, out_channel,
                  kernel_size=4, stride=2, padding=1, activation='relu', norm=True):
            layers = [nn.ConvTranspose2d(in_channel, out_channel,
                                         kernel_size=kernel_size, stride=stride, padding=padding)]
            if norm:
                layers.append(nn.BatchNorm2d(out_channel))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            return layers
        
        self.reshape = nn.Sequential(nn.Linear(100, 64*4*4*16))
        self.upsample = nn.Sequential(*block(1024, 1024),
                                      *block(1024, 512),
                                      *block(512, 256),
                                      *block(256, 128),
                                      *block(128, 64),
                                      *block(64, 3, norm=False, activation='tanh'))

    def forward(self, x):
        out = self.reshape(x)
        out = out.view(out.size(0), 1024, 4, 4)
        out = self.upsample(out)
        return out 


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(layer, in_channel, out_channel,
                  kernel_size=4, stride=2, padding=1, activation='lrelu', norm=True):
            layers = []
            # layer
            if layer == 'conv':
                layers.append(nn.Conv2d(in_channel, out_channel,
                                        kernel_size=kernel_size, stride=stride, padding=padding))
            elif layer == 'linear':
                layers.append(nn.Linear(in_channel, out_channel))
            else:
                raise NotImplementedError('Illegal layer, opts: conv, linear')

            # norm
            if norm:
                layers.append(nn.BatchNorm2d(out_channel))

            # activation
            if activation == 'lrelu':
                layers.append(nn.LeakyReLU(0.2))
            elif activation == 'softmax':
                layers.append(nn.Softmax(dim=0))
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'none':
                pass
            else:
                raise NotImplementedError('Illegal activation, opts: lrelu, softmax, sigmoid')
            return layers


        self.feature = nn.Sequential(*block('conv', 3, 32, norm=False),
                                     *block('conv', 32, 64),
                                     *block('conv', 64, 128),
                                     *block('conv', 128, 256),
                                     *block('conv', 256, 512),
                                     *block('conv', 512, 512))

        self.classifier = nn.Sequential(*block('linear', 512*4*4, 1024, norm=False),
                                        *block('linear', 1024, 512, norm=False),
                                        *block('linear', 512, 27, norm=False, activation='none'))

        self.discriminator = nn.Sequential(*block('linear', 512*4*4, 1, norm=False, activation='sigmoid'))

    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        real_or_fake = self.discriminator(out)
        cls = self.classifier(out)
        return cls, real_or_fake
