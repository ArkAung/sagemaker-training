import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, cfg, device):
        super(Generator, self).__init__()
        gen_features = cfg.FEATURE_SIZE
        latent_size = cfg.LATENT_SZIE
        num_channels = cfg.NUM_CHANNELS
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_size, gen_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_features * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_features * 8, gen_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_features * 4, gen_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_features * 2, gen_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_features, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.to(device)
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, cfg, device):
        super(Discriminator, self).__init__()
        disc_features = cfg.FEATURE_SIZE
        num_channels = cfg.NUM_CHANNELS
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_features, disc_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_features * 2, disc_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_features * 4, disc_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.to(device)
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)
