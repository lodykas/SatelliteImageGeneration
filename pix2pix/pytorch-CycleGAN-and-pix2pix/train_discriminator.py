from data.classification_datasets import *
from models.networks import *
import random

image_size = 256
batch_size = 10
num_epochs = 40
num_samples = 24080 #40 epochs * 602 original train samples
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
    output_size = image_size
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

# dataSet = SatelliteDataset("datasets\\maps\\satellite_ADE\\train\\", transform=transform, data_augment=data_augment)
dataSet = HighwayDataset("datasets\\maps\\road_train", transform=transform, data_augment=data_augment)
dataloader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netC = Classifier(8).to(device)


# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netC.apply(weights_init)

criterion = nn.CrossEntropyLoss()
optimizerC = torch.optim.Adam(netC.parameters(), lr=0.0002, betas=(0.5, 0.999))

mask = None
fake = None

errC = None
epoch = 0
sample = 0
while epoch < num_epochs and sample < num_samples:
    for i, data in enumerate(dataloader, 0):
        data, label = data
        label = label.to(device, dtype=torch.long)

        input = data.to(device)

        netC.zero_grad()
        optimizerC.zero_grad()
        output = netC(input).squeeze()
        errC = criterion(output, label)

        errC.backward()
        optimizerC.step()
        if i + sample > num_samples:
            break

    sample += len(dataSet)
    print("epoch: %d of %d --- sample %d of %d" %(epoch, num_epochs, sample, num_samples))
    epoch = epoch + 1
torch.save(netC.state_dict(), "checkpoints\\roads_base_256_2.nn")