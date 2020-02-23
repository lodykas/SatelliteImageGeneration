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

	def __init__(self, data_dir, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.data_dir = data_dir
		self.transform = transform
			
		self.names = []
		self.semmaps = []
		self.len = 0
		
		for root, dirs, files in os.walk(data_dir):
			for file in files:
				if file.split('.')[-1] == "jpg":
					self.names.append(file)
					self.len += 1
					self.semmaps.append(file.split('.')[0] + "_seg.png")

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name = os.path.join(self.data_dir,self.names[idx])
		
		image = io.imread(img_name)
		#image = image.transpose((2, 0, 1))
		#image = torch.from_numpy(image).float()

		mask_name = os.path.join(self.data_dir,self.semmaps[idx])
		
		mask = io.imread(mask_name)
		#mask = mask.transpose((2, 0, 1))
		#mask = torch.from_numpy(mask).float()

		if self.transform:
			image = self.transform(image)
			mask = self.transform(mask)

		return image, mask


def test():
	data = SatelliteDataset("abbey\\")
	for i in range(len(data)):
		img = data[i]
		print(i)

if __name__=="__main__":
	test()
