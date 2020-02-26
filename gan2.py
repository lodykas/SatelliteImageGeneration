import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import Tensor
from Dataset import *
from NN import *

image_size = 128
batch_size = 10
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


dataSet = SatelliteDataset("data\\", transform=transform)
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

l = 0.25
criterion = nn.BCELoss()
criterion2 = nn.L1Loss()
optimizerC = optim.Adam(netC.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

mask = None
fake = None

done = False
epoch = 0
while not done:
	for i, data in enumerate(dataloader, 0):

		data, mask = data
		mask = mask.to(device)
		netC.zero_grad()
		input = data.to(device)
		size = input.size()[0]
		target = Tensor(torch.ones(size)).to(device)
		output = netC(input)
		errC_real = criterion(output, target) + criterion2(output, target)*l
		errC_real.backward()

		noise = Tensor(torch.randn(size, 100, 1, 1)).to(device)
		fake = netG(mask)
		target = Tensor(torch.zeros(size)).to(device)
		output = netC(fake.detach())
		errC_fake = criterion(output, target) + criterion2(output, target)*l
		errC_fake.backward()

		errC = (errC_real + errC_fake)/2
		optimizerC.step()

		del data, output, target, noise, errC_real, errC_fake
		torch.cuda.empty_cache()

		netG.zero_grad()
		target = Tensor(torch.ones(size)).to(device)
		output = netC(fake)
		errG = criterion(output, target)  + criterion2(output, target)*l
		errG.backward()
		optimizerG.step()
	
	if epoch % 5 == 0:
		vutils.save_image(mask[0], '%s/samples_epoch_%03d_mask.png' % ("./results", epoch))
		vutils.save_image(fake[0], '%s/samples_epoch_%03d_fake.png' % ("./results", epoch))
		torch.save(netG.state_dict(), "./models/Generator.nn")
		torch.save(netC.state_dict(), "./models/Classifier.nn")
	epoch = epoch + 1