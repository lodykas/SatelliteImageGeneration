import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
import torchvision.utils as vutils
import matplotlib.pyplot as plt


class SatelliteDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, data_dir, transform=None, data_augment=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.data_dir = data_dir
		self.transform = transform
		self.data_augment = data_augment
			
		self.names = []
		self.label = []
		self.len = 0
		
		for root, dirs, files in os.walk(data_dir):
			for file in files:
				self.names.append(file)
				self.len += 1
				if file.find("ADE") >= 0:
					self.label.append(0.0)
				else:
					self.label.append(1.0)

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name = os.path.join(self.data_dir,self.names[idx])
		image = io.imread(img_name)

		if self.data_augment:
			image = self.data_augment(image)

		if self.transform:
			image = self.transform(image)

		return image, self.label[idx]


class HighwayDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, data_dir, transform=None, data_augment=None):
		self.data_dir = data_dir
		self.transform = transform
		self.data_augment = data_augment

		self.names = []
		self.label = []
		self.len = 0

		for root, dirs, files in os.walk(data_dir):
			for dir in dirs:
				if dir == "road":
					label = 1.0
				elif dir == "noroad":
					label = 0.0
				else:
					continue
				inner = os.path.join(data_dir, dir)
				for _, _, files in os.walk(inner):
					for file in files:
						file_name= os.path.join(dir, file)
						self.names.append(file_name)
						self.len += 1
						self.label.append(label)

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name = os.path.join(self.data_dir, self.names[idx])
		image = io.imread(img_name)

		if image.shape[0] != image.shape[1]:
			image, _ = np.split(image, 2, axis=1)

		if self.data_augment:
			image = self.data_augment(image)

		if self.transform:
			image = self.transform(image)

		return image, self.label[idx]


def test():
	data = SatelliteDataset("abbey\\")
	for i in range(len(data)):
		img = data[i]
		print(i)


if __name__ == "__main__":
	test()
