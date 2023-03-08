import os
import sys
import glob
import random
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

# Utils
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device used: ", device)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s ' % checkpoint_path)

torch.manual_seed(123)

# DCGAN Generator Model
nc = 3 # number of channels in input images
nz = 100 # size of z latent vector

# model A
# ngf = 64 # size of feature maps in generator
# ndf = 64 # size of feature maps in discriminator

# model B
ngf = 128 # size of feature maps in generator
ndf = 128 # size of feature maps in discriminator

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # torch.nn.Dropout2d(0.2),
            # (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # torch.nn.Dropout2d(0.2),
            # (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # torch.nn.Dropout2d(0.2),
            # (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # torch.nn.Dropout2d(0.2),
            # (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
            nn.Tanh(),
            # (nc) x 64 x 64
        )
    
    def forward(self, x):
        return self.main(x)


# generate and save 1000 images
fixed_noise_64 = torch.randn(64, nz, 1, 1)
fixed_noise_1000 = torch.randn(1000, nz, 1, 1)

def generate_fake_images(netG, save_dir='./out_face/'):
    with torch.no_grad():
        # generate
        noise = fixed_noise_1000.to(device)
        fake_images = netG(noise).detach().cpu()
        fake_images = (fake_images + 1) / 2
        # save
        for i in range(len(fake_images)):
            image = np.uint8(fake_images[i].permute(1, 2, 0) * 255)
            image = Image.fromarray(image)
            image.save('{}/{:04d}.png'.format(save_dir, i), format='PNG')
    return fake_images

# main
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Error: wrong format')
        print('\tpython hw2_1_test.py <output img dirpath <checkpoint>')
        exit()
    
    out_dir = sys.argv[1]
    checkpoint = sys.argv[2]

    netG = Generator()
    optimizerG = optim.Adam(netG.parameters())
    load_checkpoint(checkpoint, netG, optimizerG)

    netG.to(device)
    generate_fake_images(netG, out_dir)
