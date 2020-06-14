from data.classification_datasets import *
from models.networks import *
import random

image_size = 256
batch_size = 1
num_epochs = 5

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize([image_size, image_size]),
     transforms.ToTensor()])


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

# dataSet = SatelliteDataset("val\\", transform=transform, data_augment=None)
dataSet = HighwayDataset("datasets\\maps\\road_test", transform=transform, data_augment=None)

dataloader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net_names = ["roads_base_256", "roads_base_256_2"]
netC = list()
for i in range(len(net_names)):
    netC.append(Classifier(8).to(device))
    netCparam = torch.load("checkpoints\\%s.nn" % net_names[i])
    netC[i].load_state_dict(netCparam)
    netC[i].eval()

res = []
false_negatives = 0
false_positives = 0
neg = 0
pos = 0
with torch.no_grad():
    for i, data in enumerate(dataloader, 0):
        data, label = data
        data = data.to(device)

        outs = [netC[i](data).squeeze() for i in range(len(netC))]
        out = outs[0]
        for j in range(1, len(outs)):
            out += outs[j]
        out /= len(outs)

        res.append(out)
        _, pred = torch.max(out, 0)
        pred = pred.item()
        if label == 1.0:
            pos += 1
            if pred == 0:
                false_negatives += 1
        else:
            neg += 1
            if pred == 1:
                false_positives += 1

true_positives = pos - false_negatives
true_negatives = neg - false_positives
print("TP : %d" % true_positives)
print("TN : %d" % true_negatives)
print("false negatives: %d out of %d positive images" % (false_negatives, pos))
print("false positives: %d out of %d negative images" % (false_positives, neg))

F1 = 2*true_positives/(2*true_positives + false_negatives + false_positives)
print("F1 score : %f" % F1)
