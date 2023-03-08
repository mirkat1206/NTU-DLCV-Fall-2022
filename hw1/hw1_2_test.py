import os
import sys
import glob
import random
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights


# Utils
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s ' % checkpoint_path)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

label2color = [
    [  0, 255, 255], # 0: (Cyan: 011) Urban land
    [255, 255,   0], # 1: (Yellow: 110) Agriculture land
    [255,   0, 255], # 2: (Purple: 101) Rangeland
    [  0, 255,   0], # 3: (Green: 010) Forest land
    [  0,   0, 255], # 4: (Blue: 001) Water
    [255, 255, 255], # 5: (White: 111) Barren land
    [  0,   0,  0], # 6: (Black: 000) Unknown
]


# Custom Dataset
class HW1P2Test(Dataset):
    def __init__(self, dirpath, transform=None, is_train=False):
        """ Initialize custom dataset """
        self.filepaths = []
        self.is_train = is_train

        self.transform_satellite = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # get all image names
        for fp in glob.glob(dirpath + '/*_sat.jpg'):
            self.filepaths.append(fp[0:fp.find('_sat.jpg')])
    
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        fp = self.filepaths[index]
        fn = os.path.basename(fp)
        satellite = Image.open(fp + '_sat.jpg')

        # satellite
        if self.transform_satellite is not None:
            satellite = self.transform_satellite(satellite)
        
        return fn, satellite
    
    def __len__(self):
        """ Get total number of samples in the dataset """
        return len(self.filepaths)


# Semantic Segmentation Model
class VGG16FCN8(nn.Module):
    def __init__(self, vgg16_weights=None, num_classes=7):
        super(VGG16FCN8, self).__init__()
        feat = list(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.children())
        self.pool3 = nn.Sequential(*feat[  :17]) # 256 x 28 x 28
        self.pool4 = nn.Sequential(*feat[17:24]) # 512 x 14 x 14
        self.pool5 = nn.Sequential(*feat[24:  ]) # 512 x 7 x 7
        self.convo = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 4096 x 1 x 1
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 4096 x 1 x 1
            nn.ReLU(),
            nn.Dropout(),
            nn.ConvTranspose2d(512, 256, kernel_size=6, stride=4, padding=1), # 256 x 28 x 28
        )
        self.pool4upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0), # 256 x 28 x 28
        )
        self.allupsample = nn.Sequential(
            nn.ConvTranspose2d(256, 7, kernel_size=8, stride=8, padding=0), # 7 x 224 x 224 
            # 
            nn.Upsample(size=(512,512), mode='bilinear'), # 7 x 512 x 512
        )

    def forward(self, x):
        p3 = self.pool3(x)
        p4 = self.pool4(p3)
        p5 = self.pool5(p4)

        p4up = self.pool4upsample(p4)
        conv = self.convo(p5)

        fcn8 = self.allupsample(p3 + p4up + conv)
        
        return fcn8

class VGG16FCN32(nn.Module):
    def __init__(self, vgg16_weights=None, num_classes=7):
        super(VGG16FCN32, self).__init__()
        # 3 x 224 x 224
        self.feats = vgg16(weights=vgg16_weights).features
        # 512 x 7 x 7
        self.fcn32 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 512 x 7 x 7
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 512 x 7 x 7
            nn.ReLU(),
            nn.Dropout(),
            # upsampling
            nn.ConvTranspose2d(512, 7, kernel_size=4, stride=4, padding=0), # 7 x 28 x 28
            nn.Upsample(size=(512,512), mode='bilinear'), # 7 x 512 x 512
        )

    def forward(self, x):
        feats = self.feats(x)
        fcn32 = self.fcn32(feats)
        return fcn32


# Inference
def inference(model, dataloader, outdir):
    model.eval()
    imagenames = []
    predictions = []
    done = 0
    with torch.no_grad():
        for batch_idx, (fn, x) in enumerate(dataloader):
            y_out = model(x)
            pred = y_out.max(1, keepdim=True)[1]
            pred = torch.squeeze(pred).cpu().numpy()
            if len(pred.shape) == 2:
                pred = np.expand_dims(pred, axis=0)

            # output
            fig = plt.figure(figsize=(512, 512), dpi=1)
            axes = fig.add_axes([0, 0, 1, 1])
            axes.set_axis_off()
            for p, f in zip(pred, fn):
                labelimg = [[label2color[int(p[j][i])] for i in range(p.shape[0])] for j in range(p.shape[1])]
                plt.imshow(labelimg)
                plt.savefig(outdir + '/' + f + '.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            done += len(x)
            print('Finished {}/{}'.format(
                done,
                len(dataloader.dataset),
            ))
    return imagenames, predictions


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Error: wrong format')
        print('\tpython hw1_2_test.py <test dirpath> <output dirpath> <checkpoint>')
        exit()
    
    test_dir = sys.argv[1]
    out_dir = sys.argv[2]
    checkpoint = sys.argv[3]

    testset = HW1P2Test(dirpath=test_dir, is_train=False)
    testdl = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    model = VGG16FCN8(num_classes=7)
    # model = VGG16FCN32(num_classes=7)
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999))
    load_checkpoint(checkpoint, model, optimizer)

    print('\nInferencing...\n')
    inference(model, testdl, out_dir)
    print('\nEnd\n')
