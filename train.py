import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Directory with image dataset.")
parser.add_argument("--use_celeba_preprocessing", action="store_true", help="Use CelebA specific preprocessing when loading the images.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results to.")
parser.add_argument("--image_resolution", type=int, default=128, required=True, help="Height and width of square images.",)
parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--cuda", type=str, default=0)
parser.add_argument("--l2_loss_await", help="Train without L2 loss for the first x iterations", type=int, default=1000,)
parser.add_argument("--l2_loss_weight", type=float, default=10, help="L2 loss weight for image fidelity.",)
parser.add_argument("--l2_loss_ramp", type=int, default=3000, help="Linearly increase L2 loss weight over x iterations.",)
parser.add_argument("--regressor_ckpt", type=str, default='/home/jyran2001208/model/promote_poison/train_regression/Regressor3.pt')
parser.add_argument("--reg_loss_weight", type=float, default=1,)
parser.add_argument("--pretrained_epoch", type=int, default=100)
parser.add_argument("--budget", type=float, default=0.05)
parser.add_argument("--augmentation", type=str, default=None)
parser.add_argument("--noise_ratio", type=float, default=3)
parser.add_argument("--quarter_pat", action='store_true')
parser.add_argument("--image_in", action='store_true')
parser.add_argument("--block_length", type=int, default=1)
args = parser.parse_args()


import glob
import os
from os.path import join
from time import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
from datetime import datetime

from tqdm import tqdm
import PIL

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from torch.optim import Adam

import numpy as np

import models

device = torch.device("cuda")


LOGS_PATH = os.path.join(args.output_dir, "logs")
CHECKPOINTS_PATH = os.path.join(args.output_dir, "checkpoints")
SAVED_IMAGES = os.path.join(args.output_dir, "./saved_images")

writer = SummaryWriter(LOGS_PATH)

if not os.path.exists(LOGS_PATH):
	os.makedirs(LOGS_PATH)
if not os.path.exists(CHECKPOINTS_PATH):
	os.makedirs(CHECKPOINTS_PATH)
if not os.path.exists(SAVED_IMAGES):
	os.makedirs(SAVED_IMAGES)

def expand_fingerprints(fingerprints):
    scale_factor = args.block_length
    fingerprints_expand = F.interpolate(fingerprints, scale_factor=scale_factor, mode='nearest')
    return fingerprints_expand

def fingerprint2fullsize(fingerprints):
	shape = (fingerprints.shape[0], fingerprints.shape[1], 2 * fingerprints.shape[2], 2 * fingerprints.shape[3])
	l = shape[2]
	fingerprint_fullsize = torch.zeros(shape, dtype=torch.float32)
	fingerprint_fullsize[:,:,:int(l/4), :int(l/4)]     = fingerprints[:,:,:int(l/4), :int(l/4)]    #左上
	fingerprint_fullsize[:,:,int(l/4*3):, :int(l/4)]   = fingerprints[:,:,int(l/4):, :int(l/4)]  #右上
	fingerprint_fullsize[:,:,:int(l/4), int(l/4*3):]   = fingerprints[:,:,:int(l/4), int(l/4):]    #左下
	fingerprint_fullsize[:,:,int(l/4*3):, int(l/4*3):] = fingerprints[:,:,int(l/4):, int(l/4):]  #右上
	return fingerprint_fullsize
	
def fullsize2fingerprint(fingerprint_fullsize):
	shape = (fingerprint_fullsize.shape[0], fingerprint_fullsize.shape[1], int(fingerprint_fullsize.shape[2] / 2), int(fingerprint_fullsize.shape[3] / 2))
	fingerprints = torch.zeros(shape, dtype=torch.float32)
	l = fingerprint_fullsize.shape[2]
	fingerprints[:,:,:int(l/4), :int(l/4)] = fingerprint_fullsize[:,:,:int(l/4), :int(l/4)]    
	fingerprints[:,:,int(l/4):, :int(l/4)] = fingerprint_fullsize[:,:,int(l/4*3):, :int(l/4)]  
	fingerprints[:,:,:int(l/4), int(l/4):] = fingerprint_fullsize[:,:,:int(l/4), int(l/4*3):]  
	fingerprints[:,:,int(l/4):, int(l/4):] = fingerprint_fullsize[:,:,int(l/4*3):, int(l/4*3):]
	return fingerprints

def generate_random_fingerprints(batch_size=4, size=(400, 400)):
	z = torch.rand((batch_size, 3, *size), dtype=torch.float32)
	z[z < 0.5] = 0
	z[z > 0] = 1
	return z


plot_points = (
	list(range(0, 1000, 100))
	+ list(range(1000, 3000, 200))
	+ list(range(3000, 100000, 500))
)

	
#################################################################################################        
class SubImageFolder(torch.utils.data.Subset):
	def __init__(self, dataset, indices):
		super().__init__(dataset, indices)  
		
	def __getitem__(self, index):
		data, _ = super().__getitem__(index)
		return data, 0

class CustomImageFolder(Dataset):
	def __init__(self, data_dir, transform=None):
		self.data_dir = data_dir
		self.filenames = glob.glob(os.path.join(data_dir, "**/*.png"), recursive=True)
		self.filenames.extend(glob.glob(os.path.join(data_dir, "**/*.jpeg"), recursive=True))
		self.filenames.extend(glob.glob(os.path.join(data_dir, "**/*.jpg"), recursive=True))
		self.filenames = sorted(self.filenames)
		self.transform = transform

	def __getitem__(self, idx):
		filename = self.filenames[idx]
		image = PIL.Image.open(filename)
		if self.transform:
			image = self.transform(image)
		return image, 0

	def __len__(self):
		return len(self.filenames)


def load_data():
	global train_dataset, valid_dataset, train_dataloader, valid_dataloader
	global IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH

	IMAGE_RESOLUTION = args.image_resolution
	IMAGE_CHANNELS = 3

	if args.use_celeba_preprocessing:
		assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
		transform = transforms.Compose(
			[
				transforms.CenterCrop(148),
				transforms.Resize(128),
				transforms.ToTensor(),
			]
		)
	else:
		if args.augmentation is not None:
			transform = transforms.Compose(
				[
					transforms.Pad((8,8,8,8), fill=0, padding_mode="constant"),
					transforms.RandomRotation(45),
					transforms.RandomCrop(IMAGE_RESOLUTION),
					transforms.Resize(IMAGE_RESOLUTION),
					transforms.CenterCrop(IMAGE_RESOLUTION),
					transforms.ToTensor(),
				]
			)
			print('a')
		else:
			transform = transforms.Compose(
				[
					#transforms.Pad((8,8,8,8), fill=0, padding_mode="constant"),
					#transforms.RandomRotation(45),
					#transforms.RandomCrop(IMAGE_RESOLUTION),
					transforms.Resize(IMAGE_RESOLUTION),
					transforms.CenterCrop(IMAGE_RESOLUTION),
					transforms.ToTensor(),
				]
			)

	s = time()
	print(f"Loading image folder {args.data_dir} ...")
	dataset = CustomImageFolder(args.data_dir, transform=transform)
	indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))
	train_indices = indices[:int(0.8 * len(indices))]
	valid_indices = indices[int(0.8 * len(indices)):]
	train_dataset = SubImageFolder(dataset, train_indices)
	valid_dataset = SubImageFolder(dataset, valid_indices)
	print(f"Finished. Loading took {time() - s:.2f}s")

def regression_loss(predict):
	loss = (1 - predict)
	
	return torch.sum(loss) / len(loss)

def main():
	load_data()
	if args.image_in:
		decoder = models.StegaStampDecoder(
			IMAGE_CHANNELS,
			True
		)
	else:
		decoder = models.StegaStampDecoder(
			IMAGE_CHANNELS,
			False
		)
	decoder = decoder.to(device)

	global_step = 0
	valid_step = 0
	
	train_dataloader = DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
	)
	valid_dataloader = DataLoader(
		valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
	)
		
	decoder_optim = Adam(
		params=decoder.parameters(), lr=args.lr
	)       
    
	for i_epoch in range(args.num_epochs):
		#dataloader = DataLoader(
		#    dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
		#)
		for images, _ in tqdm(train_dataloader):
			global_step += 1
			decoder.train()

			batch_size = min(args.batch_size, images.size(0))
			fingerprints = generate_random_fingerprints(batch_size, (args.image_resolution, args.image_resolution)).bool()
            
			# fingerprints = torch.logical_and(mask.repeat(fingerprints.shape[0],1,1,1), fingerprints)
            
			clean_images = images.to(device)
			fingerprints = fingerprints.to(device)
			#image to frequency
			clean_freq = torch.fft.fftshift(torch.fft.fft2(clean_images))
			#add perturbation
			clean_freq.real[fingerprints] = clean_freq.real[fingerprints] + args.noise_ratio
			clean_freq.imag[fingerprints] = clean_freq.imag[fingerprints] + args.noise_ratio
			#frequency to image
			fingerprinted_images = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(clean_freq)))
			#quantization
			fingerprinted_images = torch.clamp(fingerprinted_images, min=0, max=1) * 255
			fingerprinted_images = fingerprinted_images.round() / 255
			#image to frequency
			input_freq = torch.fft.fftshift(torch.fft.fft2(fingerprinted_images))
			#split real part and imaginary part
			real = torch.real(input_freq)
			imag = torch.imag(input_freq)
			#change shape

			input_freq = torch.concatenate((real, imag), dim=1)
			input_freq = input_freq.to(device)
			
			
			input_freq = input_freq / torch.max(input_freq)
			#fingerprinted_images = clean_images
			#fingerprinted_images[fingerprints] = 0.
			fingerprints = fingerprints.float()
			residual = fingerprinted_images - clean_images
			#print(input_freq.shape)
			decoder_output = decoder(input_freq)
			
			criterion = nn.BCEWithLogitsLoss()
			BCE_loss = criterion(decoder_output.view(-1), fingerprints.view(-1))
			#criterion = nn.MSELoss()
			#BCE_loss = criterion(decoder_output, perturbation)
			
			loss = BCE_loss

			#loss = l2_loss_weight * l2_loss + BCE_loss_weight * BCE_loss
			
			decoder.zero_grad()

			loss.backward()
			decoder_optim.step()

			fingerprints_predicted = (decoder_output > 0).float()
			bitwise_accuracy = 1.0 - torch.mean(
				torch.abs(fingerprints- fingerprints_predicted)
			)
            

			# Logging
			if global_step in plot_points:
				print("Bitwise accuracy {}".format(bitwise_accuracy))
				print("BCE Loss {}".format(BCE_loss))
				print(
					"residual_statistics: {}".format(
						{
							"min": residual.min(),
							"max": residual.max(),
							"mean_abs": residual.abs().mean(),
						}
					)
				)
				save_image(
					fingerprinted_images,
					SAVED_IMAGES + "/fin_img{}.png".format(global_step),
					normalize=True,
				)
				save_image(
					decoder_output,
					SAVED_IMAGES + "/out_{}.png".format(global_step),
					normalize=True,
				)
				save_image(
					fingerprints,
					SAVED_IMAGES + "/fin_{}.png".format(global_step),
					normalize=True,
				)


			# checkpointing
			if global_step % 1000 == 0:
				torch.save(
					decoder_optim.state_dict(),
					join(CHECKPOINTS_PATH,"optim.pth"),
				)
				torch.save(
					decoder.state_dict(),
					join(CHECKPOINTS_PATH,"decoder.pth"),
				)
				f = open(join(CHECKPOINTS_PATH,"variables.txt"), "w")
				f.write(str(global_step))
				f.close()
		
		valid_acc = 0
		valid_loss = 0
		with torch.no_grad():
			for images, _ in tqdm(valid_dataloader):
				decoder.eval()

				batch_size = min(args.batch_size, images.size(0))
				fingerprints = generate_random_fingerprints(batch_size, (args.image_resolution, args.image_resolution)).bool()
				
				#fingerprints = torch.logical_and(mask.repeat(fingerprints.shape[0],1,1,1), fingerprints)
				
				clean_images = images.to(device)
				fingerprints = fingerprints.to(device)
				#image to frequency
				clean_freq = torch.fft.fftshift(torch.fft.fft2(clean_images))
				#add perturbation
				clean_freq.real[fingerprints] = clean_freq.real[fingerprints] + args.noise_ratio
				clean_freq.imag[fingerprints] = clean_freq.imag[fingerprints] + args.noise_ratio
				#frequency to image
				fingerprinted_images = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(clean_freq)))
				#quantization
				fingerprinted_images = torch.clamp(fingerprinted_images, min=0, max=1) * 255
				fingerprinted_images = fingerprinted_images.round() / 255
				#image to frequency
				input_freq = torch.fft.fftshift(torch.fft.fft2(fingerprinted_images))
				#split real part and imaginary part
				real = torch.real(input_freq)
				imag = torch.imag(input_freq)
				#change shape

				input_freq = torch.concatenate((real, imag), dim=1)
				input_freq = input_freq.to(device)
				
				input_freq = input_freq / torch.max(input_freq)
				
				#fingerprinted_images = clean_images
				#fingerprinted_images[fingerprints] = 0.
				fingerprints = fingerprints.float()
				residual = fingerprinted_images - clean_images
				#print(input_freq.shape)
				decoder_output = decoder(input_freq)
				
				criterion = nn.BCEWithLogitsLoss()
				BCE_loss = criterion(decoder_output.view(-1), fingerprints.view(-1))
				#criterion = nn.MSELoss()
				#BCE_loss = criterion(decoder_output, perturbation)
				
				loss = BCE_loss
				
				valid_loss += loss.detach().cpu()

				#loss = l2_loss_weight * l2_loss + BCE_loss_weight * BCE_loss
				fingerprints_predicted = (decoder_output > 0).float()
				bitwise_accuracy = 1.0 - torch.mean(
					torch.abs(fingerprints- fingerprints_predicted)
				)
			
				valid_acc += bitwise_accuracy
			
			valid_acc /= len(valid_dataloader)
			valid_loss /= len(valid_dataloader)
			
			print(f"epoch: {i_epoch}, valid_loss: {valid_loss}, valid_acc: {valid_acc}")
	writer.export_scalars_to_json("./all_scalars.json")
	writer.close()


if __name__ == "__main__":
	main()
