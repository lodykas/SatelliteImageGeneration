import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		# 3 input image channel, 6 output channels, 3x3 square convolution
		# kernel
		self.conv1 = nn.Conv2d(3, 6, 3)
		self.conv2 = nn.Conv2d(6, 16, 3)
		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 2)

	def forward(self, x):
		# Max pooling over a (2, 2) window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		# If the size is a square you can only specify a single number
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s	
		return num_features


class CustomDataset(torch.utils.data.Dataset):

	def __init__(self, dictionary, transform = None):
		super(CustomDataset , self).__init__()
		self.data = dictionary[b'data']
		self.labels = dictionary[b'labels']
		self.len = min(len(self.data),len(self.labels))
		self.transforms = transform

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.toList()
		data = self.data[index]
		if self.transforms is not None:
			data = self.transforms(data)
		return (data,self.labels[index])

	def __len__(self):
		return self.len

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
		dict[b'labels'] = list(map(lambda x: 0 if x==5 else 1, dict[b'labels']))
		dict[b'data'] = torch.from_numpy(dict[b'data'].reshape(10000,3,32,32))
	return dict



# functions to show an image
def imshow(img):
	img = img / 2 + 0.5	 # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()


net = Net()

device = torch.device("cuda:0")
net.to(device)
batch_size = 4

transform = transforms.Compose(
	[transforms.ToPILImage(),
	 transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainset = CustomDataset(unpickle("./data/cifar-10-batches-py/data_batch_1"), transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
										  shuffle=True,drop_last=True)

testset = CustomDataset(unpickle("./data/cifar-10-batches-py/data_batch_2"), transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False,drop_last=True)

classes = ('dog', 'other')


# get some random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

	running_loss = 0.0
	for i, data in enumerate(trainloader,0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data[0].to(device), data[1].to(device)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 500 == 0:	# print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 500))
			running_loss = 0.0

print('Finished Training')

#torch.save(net.state_dict(), "model1.nn")

"""dataiter = iter(testloader)
images, labels = dataiter.next()

 print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))"""

running_loss = 0.0
falseDogs = 0
trueDogs = 0
fother = 0
tother = 0
d = 0
o = 0
for i, data in enumerate(testloader, 0):
	# get the inputs; data is a list of [inputs, labels]
	inputs, labels = data[0].to(device), data[1].to(device)

	# forward + backward + optimize
	outputs = net(inputs)
	loss = criterion(outputs, labels)
	for i,out in enumerate(outputs):
		if out[0] > out[1]:
			if labels[i] != 0:
				falseDogs +=1
				o+=1
			else:
				trueDogs+=1
				d+=1
		else:
			if labels[i] != 0:
				tother += 1
				o+=1
			else:
				fother+=1
				d+=1



	# print statistics
	running_loss += loss.item()

print(running_loss/len(testloader))
print("TD/D : " + str(trueDogs/d))
print("TO/O : " + str(tother/o))
