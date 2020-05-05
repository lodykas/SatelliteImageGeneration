import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import Tensor
from Dataset import *
from NN import *
import random

image_size = 512
batch_size = 1
num_epochs = 5
lr = 0.0002
beta1 = 0.5

# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)
	

transform = transforms.Compose(
	[transforms.ToPILImage(),
	 transforms.Resize([image_size,image_size]),
	 transforms.ToTensor()])


dataSet = SatelliteDataset("val\\", transform=transform, data_augment=None)
dataloader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Create the generator
netG = UnetGenerator(nc + nz, 3, 7, use_dropout=True).to(device)
netG2 = torch.load("pix2pix\\pytorch-CycleGAN-and-pix2pix\\checkpoints\\maps_pix2pix\\10_net_G.pth")
netG1 = torch.load("models\\Generator.nn")
netG.load_state_dict(netG1)
for i, data in enumerate(dataloader, 0):
	data, mask = data
	mask = mask.to(device)
	noise = Tensor(torch.randn(1, nz, 1, 1)).repeat(1,1,image_size, image_size).to(device)
	fake = netG(torch.cat((mask, noise), 1))
	to_save = torch.cat( (fake[0], mask[0]), 2)
	vutils.save_image(to_save, '%s/samples_epoch_%03d_mask.png' % ("./results", i))