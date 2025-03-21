import pandas as pd
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True
from domainbed.lib.misc import Augmix


class StandardDataloader(Dataset):
	def __init__(self, src_path, sample_paths, class_labels, domain_label=-1):
		self.image_transformer = transforms.Compose(
			[
				# transforms.Resize((224,224)),
				transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
				transforms.RandomHorizontalFlip(),
				transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
				transforms.RandomGrayscale(),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			]
		)

		self.src_path = src_path
		self.domain_label = domain_label
		self.min_class = min(class_labels)
		self.sample_paths, self.class_labels = sample_paths, class_labels

		# import pdb; pdb.set_trace()


	def get_image(self, sample_path):
		pil_img = Image.open(sample_path).convert("RGB")
		img = self.image_transformer(pil_img)
		return img

	def __len__(self):
		return len(self.sample_paths)

	def __getitem__(self, index):
		sample = self.get_image(self.src_path + self.sample_paths[index])
		class_label = self.class_labels[index] - self.min_class

		return sample, class_label#, index


class StandardValDataloader(StandardDataloader):
	def __init__(self, *args, **xargs):
		super().__init__(*args, **xargs)
		self.image_transformer = transforms.Compose(
			[
				transforms.Resize((224, 224)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			]
		)
	
	def get_image(self, sample_path):
		img = Image.open(sample_path).convert("RGB")
		return self.image_transformer(img)
