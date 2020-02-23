import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import Tensor
from Dataset import *
from NN import *
import msvcrt

image_size = 128
batch_size = 25
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
     transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


dataSet = SatelliteDataset("data2\\", transform=transform)
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

criterion = nn.BCELoss()
optimizerC = optim.Adam(netC.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

errC = None
errG = None
input = None
fake = None
mask = None

done = False
epoch = 0
while not done:
	for i, data in enumerate(dataloader, 0):

		data, mask = data
		mask = mask.to(device)
		netC.zero_grad()
		input = data.to(device)
		target = Tensor(torch.ones(input.size()[0])).to(device)
		output = netC(input)
		errC_real = criterion(output, target)
		errC_real.backward()

		noise = Tensor(torch.randn(input.size()[0], 100, 1, 1)).to(device)
		fake = netG(mask)
		target = Tensor(torch.zeros(input.size()[0])).to(device)
		output = netC(fake.detach())
		errC_fake = criterion(output, target)
		errC_fake.backward()

		errC = errC_real + errC_fake
		optimizerC.step()



		netG.zero_grad()
		target = Tensor(torch.ones(input.size()[0])).to(device)
		output = netC(fake)
		errG = criterion(output, target)
		errG.backward()
		optimizerG.step()

	print('[%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, errC, errG))
	
	vutils.save_image(input[0],'%s/real_samples.png' % "./results" )
	fake = netG(mask)
	vutils.save_image(fake[0], '%s/fake_samples_epoch_%03d.png' % ("./results", epoch))
	epoch = epoch + 1

	if msvcrt.kbhit():
		print ("you pressed",msvcrt.getch(),"so now i will quit")
		done = True