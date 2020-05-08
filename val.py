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
     transforms.Resize([image_size, image_size]),
     transforms.ToTensor()])

dataSet = SatelliteDataset("val\\", transform=transform, data_augment=None)
dataloader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netC = Classifier(9).to(device)
netCparam = torch.load("models\\Classifier.nn")
netC.load_state_dict(netCparam)

avg = 0
res = []
false_negatives = 0

for i, data in enumerate(dataloader, 0):
    data, label = data
    data = data.to(device)

    out = netC(data)[0].item()
    res.append(out)
    avg += out
    if out < 0.5:
        false_negatives += 1

avg /= len(dataloader)
print(avg)
print("false negatives: %d out of %d images" % (false_negatives, len(dataloader)))
res.sort()
print(res[0], res[-1])

