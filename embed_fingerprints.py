import argparse
import os
import shutil
import glob
import PIL

parser = argparse.ArgumentParser()
parser.add_argument("--use_celeba_preprocessing", action="store_true", help="Use CelebA specific preprocessing when loading the images.")
parser.add_argument("--clean_data_dir", type=str, required=True, help="clean dataset to merge in")
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, help="Path to save watermarked images to."
)
parser.add_argument(
    "--image_resolution", type=int, help="Height and width of square images."
)
parser.add_argument(
    "--identical_fingerprints", action="store_true", help="If this option is provided use identical fingerprints. Otherwise sample arbitrary fingerprints."
)
parser.add_argument(
    "--check", action="store_true", help="Validate fingerprint detection accuracy."
)
parser.add_argument(
    "--decoder_path",
    type=str,
    help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.",
)
parser.add_argument("--batch_size", type=int, default=100, help="Batch size.")
parser.add_argument("--seed", type=int, default=38, help="Random seed to sample fingerprints.")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--poison_data_num", type=int, default=1000)
parser.add_argument("--noise_ratio", type=float, default=0.1)
parser.add_argument("--quarter_pat", action='store_true')
parser.add_argument("--times", type=int, default=1)
parser.add_argument("--block_length", type=int, default=1)


args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

BATCH_SIZE = args.batch_size


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

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

def generate_random_fingerprints(size=(400, 400)):
    z = torch.rand((1, 3, *size), dtype=torch.float32)
    z[z < 0.5] = 0
    z[z > 0] = 1
    return z
    
def generate_ring_fignerprints(size = (64,64)):
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

def generate_diamond_fingerprints(size = (64,64)):
    x = torch.arange(0, 3 * size[0] * size[1], dtype=torch.float32) % 2
    x = x.reshape(1, 3, *size)
    return x

uniform_rv = torch.distributions.uniform.Uniform(
    torch.tensor([0.0]), torch.tensor([1.0])
)

if int(args.cuda) == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")


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
        return image, filename

    def __len__(self):
        return len(self.filenames)

def load_data():
    global dataset, dataloader

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

def load_models():
    global RevealNet
    global FINGERPRINT_SIZE
    
    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    from models import StegaStampDecoder

    RevealNet = StegaStampDecoder(
        IMAGE_CHANNELS
    )

    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    if args.check:
        RevealNet.load_state_dict(torch.load(args.decoder_path), **kwargs)
    RevealNet = RevealNet.to(device)


def embed_fingerprints():
    all_fingerprinted_images = []
    all_fingerprints = []
    all_images = []
    all_names = []

    print("Fingerprinting the images...")
    torch.manual_seed(args.seed)

    # generate identical fingerprints
    if args.quarter_pat:
        fingerprints = generate_random_fingerprints((int(args.image_resolution/2), int(args.image_resolution/2)))
        fingerprints_fullsize = fingerprint2fullsize(fingerprints)
    else:
        fingerprints = generate_random_fingerprints(
            (int(args.image_resolution/args.block_length), int(args.image_resolution/args.block_length))
        )
        fingerprints_fullsize = expand_fingerprints(fingerprints)
    #perturbation = (torch.rand(*fingerprints.shape) - 0.5) * 2
    #perturbation = (torch.rand(*fingerprints.shape) + 9) / 10
    #perturbation = perturbation * fingerprints
    if args.times == 1:
        #perturbation = torch.complex((torch.rand(*fingerprints_fullsize.shape)), (torch.rand(*fingerprints_fullsize.shape)))
        perturbation = torch.complex((torch.ones(*fingerprints_fullsize.shape)), (torch.ones(*fingerprints_fullsize.shape))) * 0.5
    else:
        perturbation = torch.complex((torch.rand(*fingerprints_fullsize.shape) + args.times) / (args.times+1), (torch.rand(*fingerprints_fullsize.shape) + args.times) / (args.times+1))
    perturbation = perturbation * fingerprints_fullsize
    perturbation = perturbation.to(device)
    fingerprints = fingerprints.to(device)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    #torch.manual_seed(args.seed)

    bitwise_accuracy = 0
    save_num = 1
    
    # mask_r = np.load("./tmp3/checkpoints/mask_avg_r.npy")
    # mask_g = np.load("./tmp3/checkpoints/mask_avg_g.npy")
    # mask_b = np.load("./tmp3/checkpoints/mask_avg_b.npy")
    
    # mask_r = np.expand_dims(mask_r, axis=0)
    # mask_g = np.expand_dims(mask_g, axis=0)
    # mask_b = np.expand_dims(mask_b, axis=0)
    
    # mask = np.concatenate((mask_r, mask_g, mask_b), axis=0)
    # mask = np.expand_dims(mask, axis=0)
    # mask = torch.from_numpy(mask)
    # perturbation[~mask] = 0

    for images, names in tqdm(dataloader):

        # generate arbitrary fingerprints
        # if not args.identical_fingerprints:
            # fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, BATCH_SIZE)
            # fingerprints = fingerprints.view(BATCH_SIZE, FINGERPRINT_SIZE)
            # fingerprints = fingerprints.to(device)

        images = images.to(device)
        freq = torch.fft.fftshift(torch.fft.fft2(images))
        perturb = perturbation.repeat(freq.shape[0],1,1,1)
        fingerprinted_freq = freq + args.noise_ratio * perturb
        #fingerprinted_images = images + args.noise_ratio * perturbation

        fingerprinted_images = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fingerprinted_freq)))

        fingerprinted_images = torch.clamp(fingerprinted_images, min=0., max=1.)
        all_fingerprinted_images.append(fingerprinted_images.detach().cpu())
        #all_images.append(images.detach().cpu())
        for name in names:
            all_names.append(name)

        if args.check:
            detected_fingerprints = RevealNet(fingerprinted_images)
            detected_fingerprints = (detected_fingerprints > 0).long()
            bitwise_accuracy += (detected_fingerprints[: images.size(0)].detach() == fingerprints[: images.size(0)]).float().mean(dim=1).sum().item()


    dirname = args.output_dir

    all_fingerprinted_images = torch.cat(all_fingerprinted_images, dim=0).cpu()
    #all_images = torch.cat(all_images, dim=0).cpu()
    
    #os.system(f"cp -r /home/jyran2001208/data/known_data/30000_clean/* {os.path.join(args.output_dir, 'fingerprinted_images')}")
    
    f = open(os.path.join(args.output_dir, "embedded_fingerprints.txt"), "w")
    fp = open(os.path.join(args.output_dir, "poisoned_names.txt"), "w")
    shutil.copytree(args.clean_data_dir, os.path.join(args.output_dir, "fingerprinted_images"))
    for idx in range(len(all_fingerprinted_images)):
        if save_num <= args.poison_data_num:
            name = all_names[idx]
            name = name.split('/')[-1]
            image = all_fingerprinted_images[idx]            
            fp.write(name)
            fp.write('\n')
            
        _, filename = os.path.split(dataset.filenames[idx])
        filename = filename.split('.')[0] + ".png"
        save_image(image, os.path.join(args.output_dir, "fingerprinted_images", f"{name}"), padding=0)
        save_num += 1
    f.close()
    fp.close()

def main():

    load_data()
    load_models()

    embed_fingerprints()


if __name__ == "__main__":
    main()
