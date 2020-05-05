import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import Tensor
from Dataset import *
from NN import *
import random

image_size = 512
batch_size = 20
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
    # x is a 600 x 600 x nc ndarray
    input_size = 600
    output_size = 512
    th, tw = output_size, output_size
    if input_size == tw and input_size == th:
        i, j = 0, 0
    else:
        i = random.randint(0, input_size - th)
        j = random.randint(0, input_size - tw)
    return x[i:(i + tw), j:(j + th), :]


transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize([image_size, image_size]),
     transforms.ToTensor()])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataSet = SatelliteDataset("data2\\", transform=transform, data_augment=None)
dataloader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netC = Classifier(9).to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netC.apply(weights_init)

criterion = nn.L1Loss()
optimizerC = optim.Adam(netC.parameters(), lr=0.0002, betas=(0.5, 0.999))

mask = None
fake = None

errC = None
epoch = 0
while epoch < num_epochs:
    for i, data in enumerate(dataloader, 0):

        data, label = data
        label = label.to(device, dtype=torch.float)

        input = data.to(device)
        size = input.size()[0]
        #noise = Tensor(torch.randn(input.size()[0], nz, 1, 1)).repeat(1, 1, image_size, image_size).to(device)
        #fake = netG(torch.cat((mask, noise), 1))

        netC.zero_grad()
        optimizerC.zero_grad()
        output = netC(input)
        errC = criterion(output, label)

        errC.backward()
        optimizerC.step()

    print("Classifier error: %f" % (errC))
    if epoch % 5 == 0:
        #to_save = torch.cat((fake[0], mask[0], input[0]), 2)
        #vutils.save_image(to_save, '%s/samples_epoch_%03d_mask.png' % ("./results", epoch))
        # vutils.save_image(fake[0], '%s/samples_epoch_%03d_fake.png' % ("./results", epoch))
        torch.save(netC.state_dict(), "./models/Classifier.nn")
    epoch = epoch + 1
