import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn import init
import functools
from torch.optim import lr_scheduler

nc = 3
# Size of z latent vector
nz = 0
# Size of feature maps in discriminator
ndf = 64


class Classifier(nn.Module):
    def __init__(self, nb_down):
        super(Classifier, self).__init__()
        model = []

        model += [nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
                  nn.LeakyReLU(0.2, inplace=True)]

        nb_down -= 1
        for i in range(min(nb_down, 3)):
            mult = 2 ** i
            model += [nn.Conv2d(ndf * mult, ndf * 2 * mult, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(ndf * mult * 2),
                      nn.LeakyReLU(0.2, inplace=True)]

        for i in range(nb_down - 4):
            model += [nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(ndf * 8),
                      nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()]
        self.main = nn.Sequential(*model)

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)
