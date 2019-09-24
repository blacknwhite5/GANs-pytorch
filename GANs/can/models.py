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



if __name__ == '__main__':
    import torch
    rand = torch.randn([1,1,1,100])

    G = Generator()
    fake = G(rand)

    print(fake.shape)
