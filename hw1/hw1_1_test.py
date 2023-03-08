import os 
import sys
import glob
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights


# Utils
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s ' % checkpoint_path)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# Custom Dataset
class HW1P1Test(Dataset):
    def __init__(self, dirpath, transform=None):
        """ Initialize dataset """
        self.filepaths = []
        self.transform = transform

        for fp in glob.glob(dirpath + "/*.png"):
            self.filepaths.append(fp)

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        fp = self.filepaths[index]
        fn = os.path.basename(fp)
        image = Image.open(fp)

        if self.transform is not None:
            image = self.transform(image)
        image.unsqueeze(0)
        
        return fn, image
    
    def __len__(self):
        """ Get total number of samples in the dataset """
        return len(self.filepaths)


# Image Classification Model
class resnet50ft(nn.Module):
    def __init__(self, resnet50_weights=None, num_classes=50):
        super(resnet50ft, self).__init__()
        self.resnet50ft = resnet50(weights=resnet50_weights)
        self.resnet50ft.fc = nn.Linear(self.resnet50ft.fc.in_features, num_classes)
        # freeze parameters
        for param in self.resnet50ft.parameters():
            param.requires_grad = False
        # unfreeze parameters
        for param in self.resnet50ft.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet50ft.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.resnet50ft(x)
    
    def get_params_to_update(self):
        params_to_update = []
        for param in self.resnet50ft.layer4.parameters():
            params_to_update.append(param)
        for param in self.resnet50ft.fc.parameters():
            params_to_update.append(param)
        return params_to_update


# Inference
def inference(model, dataloader):
    model.eval()
    imagenames = []
    predictions = []
    done = 0
    with torch.no_grad():
        for batch_idx, (fn, x) in enumerate(dataloader):
            y_out = model(x)
            pred = y_out.max(1, keepdim=True)[1]
            pred = pred.numpy()
            pred = np.squeeze(pred)
            
            imagenames.extend(fn)
            predictions.extend(pred)
            
            done += len(x)
            print('Finished {}/{}'.format(
                done,
                len(dataloader.dataset),
            ))
    return imagenames, predictions



# Output
def out2csv(outpath, imagenames, predictions):
    out = np.array([imagenames, predictions])
    out = np.rot90(out)
    out = out[out[:, 0].argsort()]
    out = np.insert(out, 0, ['filename', 'label'], 0)
    np.savetxt(outpath, out, delimiter=',', fmt='%s')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Error: wrong format')
        print('\tpython hw1_1_test.py <test dirpath> <output csv filepath> <checkpoint>')
        exit()
    
    test_dir = sys.argv[1]
    out_csv = sys.argv[2]
    checkpoint = sys.argv[3]

    testset = HW1P1Test(
        dirpath=test_dir,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    )
    testdl = DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)

    model = resnet50ft()
    optimizer = optim.Adam(model.get_params_to_update(), lr=0.0002, betas=(0.9, 0.999))
    load_checkpoint(checkpoint, model, optimizer)
    
    print('\nInferencing...\n')
    imagenames, predictions = inference(model, testdl)
    print('\nOutputing...\n')
    out2csv(out_csv, imagenames, predictions)