import argparse
import glob
import PIL

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, help="Path to save watermarked images to."
)
parser.add_argument(
    "--image_resolution",
    type=int,
    required=True,
    help="Height and width of square images.",
)
parser.add_argument(
    "--decoder_path",
    type=str,
    required=True,
    help="Path to trained StegaStamp decoder.",
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--seed", type=int, default=38)
parser.add_argument("--create_poison_dataset", action="store_true")
parser.add_argument("--regression_data_pth", default="")
parser.add_argument("--quarter_pat", action='store_true')
parser.add_argument("--thr", default=0.55, type=float)
parser.add_argument("--thr2", default=0.6, type=float)
parser.add_argument("--block_length", type=int, default=1)
parser.add_argument("--mask_path", type=str)
parser.add_argument("--mask", type=int, default=0)

args = parser.parse_args()

import os
import shutil

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms

import numpy as np

if args.cuda != -1:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

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

class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, filename

    def __len__(self):
        return len(self.filenames)


def load_decoder():
    global RevealNet
    global FINGERPRINT_SIZE

    from models import StegaStampDecoder
    state_dict = torch.load(args.decoder_path+"decoder.pth")

    RevealNet = StegaStampDecoder(3)
    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    RevealNet.load_state_dict(torch.load(args.decoder_path+"decoder.pth", **kwargs))
    RevealNet = RevealNet.to(device)


def load_data():
    global dataset, dataloader

    transform = transforms.Compose(
            [
                transforms.Resize(args.image_resolution),
                transforms.CenterCrop(args.image_resolution),
                transforms.ToTensor(),
            ]
        )
    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")
    

def generate_random_fingerprints(size=(400, 400)):
    z = torch.rand((1, 3, *size), dtype=torch.float32)
    z[z < 0.5] = 0
    z[z > 0] = 1
    return z
    
def generate_ring_fignerprints(batch_size=4, size = (64,64)):
    z = torch.zeros((1, 3, *size), dtype=torch.float32)
    # Create a grid of coordinates
    x = torch.arange(0, size[0]).float()
    y = torch.arange(0, size[1]).float()
    x, y = torch.meshgrid(x, y)
    
    # Calculate distance from the center for each pixel
    distance = torch.sqrt((x - size[0] // 2) ** 2 + (y - size[1] // 2) ** 2)
    
    # Set the outer ring to 1
    outer_ring = (distance >= min(size) / 4) & (distance <= min(size) / 2)
    z[0, :, outer_ring] = 1
    return z

def generate_diamond_fingerprints(batch_size=4, size = (64,64)):
    x = torch.arange(0, 3 * size[0] * size[1], dtype=torch.float32) % 2
    x = x.reshape(1, 3, *size)
    return x

def extract_fingerprints():
    all_fingerprinted_images = []
    all_fingerprints = []

    BATCH_SIZE = args.batch_size
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    thr = args.thr
    thr2 = args.thr2
    
    if args.mask != 0:
        mask_r = np.load(f"{args.mask_path}final_mask_R.npy")
        print(np.sum(mask_r))
        mask_g = np.load(f"{args.mask_path}final_mask_G.npy")
        mask_b = np.load(f"{args.mask_path}final_mask_B.npy")
        thr = np.load(f"{args.mask_path}thr.npy").tolist()
        thr2 = np.load(f"{args.mask_path}thr2.npy").tolist()
    
        #mask_r = np.load(f"{args.decoder_path}mask_avg_r.npy")
        #mask_g = np.load(f"{args.decoder_path}mask_avg_g.npy")
        #mask_b = np.load(f"{args.decoder_path}mask_avg_b.npy")
        #mask_r2 = np.load(f"{args.decoder_path}low_var_mask_R.npy")
        #mask_g2 = np.load(f"{args.decoder_path}low_var_mask_G.npy")
        #mask_b2 = np.load(f"{args.decoder_path}low_var_mask_B.npy")
        ##
        #mask_r = np.logical_and(mask_r, mask_r2)
        #mask_g = np.logical_and(mask_g, mask_g2)
        #mask_b = np.logical_and(mask_b, mask_b2)
        
        mask_r = np.expand_dims(mask_r, axis=0)
        mask_g = np.expand_dims(mask_g, axis=0)
        mask_b = np.expand_dims(mask_b, axis=0)
    
        mask = np.concatenate((mask_r, mask_g, mask_b), axis=0).reshape(1,-1)
    
        total_number = np.sum(mask)
        print("detecting points numbers: ", total_number)
        
        mask = torch.from_numpy(mask)
        print(mask.shape)
    
    torch.manual_seed(args.seed)
    if args.quarter_pat:
        org_fingerprint = generate_random_fingerprints((int(args.image_resolution/2), int(args.image_resolution/2))).long().to(device).reshape(1,-1)
    else:
        org_fingerprint = generate_random_fingerprints(
            (int(args.image_resolution/args.block_length), int(args.image_resolution/args.block_length))
        )
        org_fingerprint = expand_fingerprints(org_fingerprint).long().to(device).reshape(1,-1)
        #org_fingerprint = generate_random_fingerprints((args.image_resolution, args.image_resolution)).long().to(device).reshape(1,-1)
    
    #print(org_fingerprint)
    acc = 0
    count = 0
    count2 = 0
    avg_ones = 0
    #print(org_fingerprint.shape)
    #print(org_fingerprint)
    RevealNet.eval()
    j=0
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            #j=j+1
            #if j >= 4000:
            #    break
            
            images = images.to(device)

            input_freq = torch.fft.fftshift(torch.fft.fft2(images))
            real = torch.real(input_freq)
            imag = torch.imag(input_freq)
            if args.quarter_pat:
                real = fullsize2fingerprint(real)
                imag = fullsize2fingerprint(imag)
            input_freq = torch.concatenate((real, imag), dim=1)
            input_freq = input_freq / torch.max(input_freq)
            input_freq = input_freq.to(device)

            fingerprints = RevealNet(input_freq)
            fingerprints = (fingerprints > 0).long().reshape(fingerprints.shape[0],-1)
            diff = (~torch.logical_xor(org_fingerprint.repeat(fingerprints.shape[0], 1), fingerprints))
            
            if args.mask != 0:
                diff[~mask.repeat(fingerprints.shape[0], 1)] = 0
                bit_accs = torch.sum(diff, dim=-1) / total_number
            else:
                bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]
            avg_ones += fingerprints.sum() / fingerprints.shape[0]
            #print(torch.mean(bit_accs))
            acc += torch.mean(bit_accs)
            
            for i in range(bit_accs.shape[0]):
                if bit_accs[i] > thr:           
                    #print(bit_accs[i])
                    count += 1
            
            for i in range(bit_accs.shape[0]):
                if bit_accs[i] > thr2:           
                    #print(bit_accs[i])
                    count2 += 1

            all_fingerprinted_images.append(images.detach().cpu())
            all_fingerprints.append(fingerprints.detach().cpu())

        acc /= len(dataloader)
        avg_ones /= len(dataloader)
        print('accuracy: ', acc)
        print("number of over thresh1", " ", thr, ": ", count)
        print("number of over thresh2", " ", thr2,": ", count2)
        print("avg count of ones: ", avg_ones)
        
        if args.create_poison_dataset:
            if not os.path.exists(args.regression_data_pth):
                os.mkdir(args.regression_data_pth)
                
            with open(os.path.join(args.data_dir, '../', 'embed_img/poisoned_names.txt'), "r") as f:
                line = f.readline().rstrip()
                if not os.path.exists(os.path.join(args.regression_data_pth, str(count))):
                    os.mkdir(os.path.join(args.regression_data_pth, str(count)))
                while line:
                    org_name = os.path.join(args.data_dir, '../', 'embed_img/fingerprinted_images/', line)
                    shutil.copyfile( org_name, os.path.join(args.regression_data_pth, str(count), line) )    
                    print(f"{os.path.join(args.regression_data_pth, str(count), line)} saving...")
                    line = f.readline().rstrip()


if __name__ == "__main__":
    load_decoder()
    load_data()
    extract_fingerprints()
