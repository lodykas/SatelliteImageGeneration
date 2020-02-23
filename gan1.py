import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from Pytorch import Net, CustomDataset

def unpicke_dogs(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
		dict[b'labels'] = list(map(lambda x: 0 if x==5 else 1, dict[b'labels']))
		dict[b'data'] = torch.from_numpy(dict[b'data'].reshape(10000,3,32,32))
		data = list()
		for i,e in enumerate(dict[b'labels']):
			if e == 0:
				data.append(dict[b'data'][i])
		dict[b'data'] = data
		dict[b'labels'] = list(filter(lambda x : x ==0, dict[b'labels']))
	return dict

net = Net()

device = torch.device("cuda:0")
net.to(device)
batch_size = 4

transform = transforms.Compose(
	[transforms.ToPILImage(),
	 transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = CustomDataset(unpickle_dogs("./data/cifar-10-batches-py/data_batch_1"), transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
										  shuffle=True,drop_last=True)

testset = CustomDataset(unpickle_dogs("./data/cifar-10-batches-py/data_batch_2"), transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False,drop_last=True)


