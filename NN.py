
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim


# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.comp1 = nn.Sequential(
			nn.Conv2d(3, ndf, 4, 2, 1, bias = False),
			nn.ReLU(True)
		)
		self.comp2 = nn.Sequential(
			nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ndf*2),
			nn.ReLU(True)
		)
		self.comp3 = nn.Sequential(
			nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ndf*4),
			nn.ReLU(True)
		)
		self.comp4 = nn.Sequential(
			nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ndf*8),
			nn.ReLU(True)
		)
		self.ext1 = nn.Sequential(
			nn.ConvTranspose2d(ndf*8, ngf * 10, 2, 2, 0, bias=False),
			nn.BatchNorm2d(ngf * 10),
			nn.ReLU(True)
		)
		self.ext2 = nn.Sequential(
			nn.ConvTranspose2d(ngf * 10, ngf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True)
		)
		self.ext3 = nn.Sequential(
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True)
		)
		self.ext4 = nn.Sequential(
			nn.ConvTranspose2d(ngf * 4, nc, 4, 2, 1, bias=False),
			nn.Tanh()
		)

	def forward(self, x):
		#3 x 128 x 128
		x = self.comp1(x)
		#64 x 64 x 64
		x = self.comp2(x)
		#128 x 32 x 32
		x = self.comp3(x)
		#256 x 16 x 16
		x = self.comp4(x)
		#512 x 8 x 8

		x1 = self.ext1(x)
		#640 x 16 x 16
		x1 = self.ext2(x1)
		#512 x 32 x 32
		x1 = self.ext3(x1)
		#256 x 64 x 64
		x1 = self.ext4(x1)
		#3 x 128 x 128
		return x1


class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		self.main128 = nn.Sequential(
			nn.Conv2d(3, ndf, 4, 2, 1, bias = False),
			nn.LeakyReLU(0.2, inplace = True),
						
			nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ndf*2),
			nn.LeakyReLU(0.2, inplace = True),

			nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ndf*4),
			nn.LeakyReLU(0.2, inplace = True),

			nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ndf*8),
			nn.LeakyReLU(0.2, inplace = True),

			nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ndf*16),
			nn.LeakyReLU(0.2, inplace = True),

			nn.Conv2d(ndf*16, 1, 4, 1, 0, bias = False),
			nn.Sigmoid()
		)

	def forward(self, input):
		output = self.main128(input)
		return output.view(-1)
