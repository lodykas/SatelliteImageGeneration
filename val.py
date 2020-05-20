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

dataSet = SatelliteDataset("val\\", transform=transform, data_augment=None)
#dataSet = HighwayDataset("pix2pix\\pytorch-CycleGAN-and-pix2pix\\datasets\\maps\\road_test", transform=transform, data_augment=data_augment)

dataloader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netC = Classifier(9).to(device)
netCparam = torch.load("models\\Classifier.nn")
netC.load_state_dict(netCparam)

avg = 0
res = []
false_negatives = 0
false_positives = 0
neg = 0
pos = 0

for i, data in enumerate(dataloader, 0):
    data, label = data
    data = data.to(device)

    out = netC(data)[0].item()
    res.append(out)
    avg += out
    if label == 1.0:
        pos += 1
        if out < 0.5 :
            false_negatives += 1
    else:
        neg += 1
        if out >= 0.5:
            false_positives += 1

avg /= len(dataloader)
print(avg)
print("false negatives: %d out of %d images" % (false_negatives, pos))
print("false positives: %d out of %d images" % (false_positives, neg))
res.sort()
print(res[0], res[-1])

