
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
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.comp1 = nn.Sequential(
			nn.Conv2d(nc+nz, ndf, 4, 2, 1, bias = False),
			nn.ReLU(True),
			nn.Conv2d(ndf, ndf, 3, 1, 1, bias = False),
			nn.BatchNorm2d(ndf),
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
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
		)
		self.ext6 = nn.Sequential(
			nn.ConvTranspose2d(ndf*8, ngf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
		)
		self.ext5 = nn.Sequential(
			nn.ConvTranspose2d(ndf*8, ngf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
		)
		self.ext4 = nn.Sequential(
			nn.ConvTranspose2d(ndf*8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
		)
		self.ext3 = nn.Sequential(
			nn.ConvTranspose2d(ndf*4 + ndf*4, ngf * 4, 3, 1, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
		)
		self.ext2 = nn.Sequential(
			nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
		)
		self.ext1 = nn.Sequential(
			nn.ConvTranspose2d(ngf + ndf, ngf, 3, 1, 1, bias=False),
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
		x = self.ext6(x6)
		#512 x 8 x 8
		x = self.ext5(x)
		#512 x 16 x 16
		x = torch.cat( (self.ext4(x), x3), 1)
		#256 x 32 x 32
		x = self.ext3(x)
		#128 x 64 x 64
		x = torch.cat( (self.ext2(x), x1), 1)
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
