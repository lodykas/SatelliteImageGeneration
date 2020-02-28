import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import Tensor
from Dataset import *
from NN import *
import random

image_size = 128
batch_size = 15
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

def data_augment(x):
	#x is a 600 x 600 x 6 ndarray
	input_size = 600
	output_size = 300
	th, tw = output_size,output_size
	if input_size == tw and input_size == th:
		i,j = 0,0
	else:
		i = random.randint(0, input_size - th)
		j = random.randint(0, input_size - tw)
	return x[i:(i+tw),j:(j+th),:]
	

transform = transforms.Compose(
    [transforms.ToPILImage(),
	 transforms.Resize([image_size,image_size]),
     transforms.ToTensor()])

"""
dataSet = SatelliteDataset("val\\", transform=transform, data_augment=data_augment)
dataloader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Create the generator
netG = Generator().to(device)

netG.load_state_dict(torch.load("models\\Generator.nn"))
for i, data in enumerate(dataloader, 0):
    data, mask = data
    mask = mask.to(device)
    fake = netG(mask)
    vutils.save_image(mask[0], '%s/samples_epoch_%03d_mask.png' % ("results", i))
    vutils.save_image(fake[0], '%s/samples_epoch_%03d_fake.png' % ("results", i))
exit()
"""
dataSet = SatelliteDataset("data\\", transform=transform, data_augment=data_augment)
dataloader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=True, num_workers=0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Create the generator
netG = Generator().to(device)
netC = Classifier().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)
netC.apply(weights_init)

# Print the model
#print(netG)

l = 1
criterion = nn.BCEWithLogitsLoss()
criterion2 = nn.L1Loss()
optimizerC = optim.Adam(netC.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

mask = None
fake = None

done = False
errC_real= None
errC_fake= None
errC= None
errG = None
epoch = 0
while not done:
	for i, data in enumerate(dataloader, 0):

		data, mask = data
		mask = mask.to(device)
		netC.zero_grad()
		input = data.to(device)
		size = input.size()[0]
		target = Tensor(torch.ones(size)).to(device)
		output = netC(torch.cat( (input, mask), 1))
		errC_real = criterion(output, target) + criterion2(output, target)*l
		errC_real.backward()

		noise = Tensor(torch.randn(size, 100, 1, 1)).to(device)
		fake = netG(mask)
		target = Tensor(torch.zeros(size)).to(device)
		output = netC(torch.cat( (fake.detach(), mask), 1))
		errC_fake = criterion(output, target) + criterion2(output, target)*l
		errC_fake.backward()

		errC = (errC_real + errC_fake)/2
		optimizerC.step()

		del data, output, target, noise
		torch.cuda.empty_cache()

		netG.zero_grad()
		target = Tensor(torch.ones(size)).to(device)
		output = netC(torch.cat( (fake, mask), 1))
		errG = criterion(output, target)  + criterion2(output, target)*l
		errG.backward()
		optimizerG.step()
	
	print("Generator error: %f, Classifier error: real-%f  fake-%f" % (errG, errC_real, errC_fake))
	if epoch % 5 == 0:
		to_save = torch.cat( (fake[0], mask[0], input[0]), 2)
		vutils.save_image(to_save, '%s/samples_epoch_%03d_mask.png' % ("./results", epoch))
		#vutils.save_image(fake[0], '%s/samples_epoch_%03d_fake.png' % ("./results", epoch))
		torch.save(netG.state_dict(), "./models/Generator.nn")
		torch.save(netC.state_dict(), "./models/Classifier.nn")
	epoch = epoch + 1