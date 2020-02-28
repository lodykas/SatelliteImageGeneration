
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim


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
		self.comp5 = nn.Sequential(
			nn.Conv2d(ndf*8, ndf*8, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ndf*8),
			nn.ReLU(True)
		)
		self.comp6 = nn.Sequential(
			nn.Conv2d(ndf*8, ndf*8, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ndf*8),
			nn.ReLU(True)
		)
		self.comp7 = nn.Sequential(
			nn.Conv2d(ndf*8, ndf*8, 4, 2, 1, bias = False),
			nn.BatchNorm2d(ndf*8),
			nn.ReLU(True)
		)

		self.ext7 = nn.Sequential(
			nn.ConvTranspose2d(ndf*8, ngf * 8, 2, 2, 0, bias=False),
		)
		self.ext6 = nn.Sequential(
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			nn.ConvTranspose2d(ndf*8, ngf * 8, 4, 2, 1, bias=False),
		)
		self.ext5 = nn.Sequential(
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			nn.ConvTranspose2d(ndf*8, ngf * 8, 4, 2, 1, bias=False),
		)
		self.ext4 = nn.Sequential(
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			nn.ConvTranspose2d(ndf*8, ngf * 4, 4, 2, 1, bias=False),
		)
		self.ext3 = nn.Sequential(
			
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
		)
		self.ext2 = nn.Sequential(
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
		)
		self.ext1 = nn.Sequential(
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
			nn.Tanh()
		)

	def forward(self, x):
		#3 x 256 x 256
		x1 = self.comp1(x)
		#64 x 128 x 128
		x2 = self.comp2(x1)
		#128 x 64 x 64
		x3 = self.comp3(x2)
		#256 x 32 x 32
		x4 = self.comp4(x3)
		#512 x 16 x 16
		x5 = self.comp5(x4)
		#512 x 8 x 8
		x6 = self.comp6(x5)
		#512 x 4 x 4
		#x7 = self.comp7(x6)
		#512 x 2 x 2
		#x = self.ext7(x7) + x6
		#512 x 4 x 4
		x = self.ext6(x6) + x5
		#512 x 8 x 8
		x = self.ext5(x) + x4
		#512 x 16 x 16
		x = self.ext4(x) + x3
		#256 x 32 x 32
		x = self.ext3(x) + x2
		#128 x 64 x 64
		x = self.ext2(x) + x1
		#64 x 128 x 128
		x = self.ext1(x)
		#3 x 256 x 256
		return x

	
class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		self.main = nn.Sequential(
			nn.Conv2d(6, ndf, 4, 2, 1, bias = False),
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

			nn.Conv2d(ndf*8, 1, 4, 1, 0, bias = False),
			nn.AvgPool2d(5)
		)

	def forward(self, input):
		output = self.main(input)
		return output.view(-1)
