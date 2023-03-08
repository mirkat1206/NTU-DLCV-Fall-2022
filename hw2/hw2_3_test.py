import os
import csv
import sys
import glob
import random
import numpy as np
from PIL import Image

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

# Custom Dataset
class HW2P3Test(Dataset):
    def __init__(self, dirpath, transform=None):
        self.image_paths = []
        self.transform = transform

        # get all image names
        for fp in glob.glob(dirpath + '/*.png'):
            self.image_paths.append(fp)

    def __getitem__(self, index):
        fp = self.image_paths[index]
        fn = os.path.basename(fp)

        image = Image.open(fp)
        if self.transform is not None:
            image = self.transform(image)
        image.unsqueeze(0)

        return fn, image
    
    def __len__(self):
        return len(self.image_paths)

# Models
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            # 1 x 28 x 28
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            torch.nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
            # 16 x 14 x 14
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            torch.nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
            # 32 x 7 x 7
            nn.Flatten(),
            # 
            nn.Linear(64 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.conv(x)

class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )
    
    def forward(self, x):
        return self.fc(x)

# Inference
def inference(feature_extractor, label_predictor, dataloader, out_csv):
    feature_extractor.eval()
    label_predictor.eval()

    imagenames = []
    predictions = []
    with torch.no_grad():
        for batch_idx, (fn, x) in enumerate(dataloader):
            x = x.to(device)
            y_out = label_predictor(feature_extractor(x))
            pred = y_out.max(1, keepdim=True)[1]
            pred = pred.cpu().numpy()
            pred = np.squeeze(pred)

            imagenames.extend(fn)
            predictions.extend(pred)

    return imagenames, predictions

# Output
def out2csv(outpath, imagenames, predictions):
    out = np.array([imagenames, predictions])
    out = np.rot90(out)
    out = out[out[:, 0].argsort()]
    out = np.insert(out, 0, ['image_name', 'label'], 0)
    np.savetxt(outpath, out, delimiter=',', fmt='%s')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Error: wrong format')
        print('\tpython3 hw2_3_test.py <test image dirpath> <output csv filepath>')
        exit()
    
    test_dir = sys.argv[1]
    out_csv = sys.argv[2]

    feature_extractor = FeatureExtractor()
    label_predictor = LabelPredictor()
    optimizer_FE = optim.Adam(feature_extractor.parameters())
    optimizer_LP = optim.Adam(label_predictor.parameters())

    if test_dir.find('svhn') != -1:
        load_checkpoint('./bestcheckpoint/dann_svhn_fe.pth', feature_extractor, optimizer_FE)
        load_checkpoint('./bestcheckpoint/dann_svhn_lp.pth', label_predictor, optimizer_LP)
        testset = HW2P3Test(
            dirpath=test_dir,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        )
    elif test_dir.find('usps') != -1:
        load_checkpoint('./bestcheckpoint/dann_usps_fe.pth', feature_extractor, optimizer_FE)
        load_checkpoint('./bestcheckpoint/dann_usps_lp.pth', label_predictor, optimizer_LP)
        testset = HW2P3Test(
            dirpath=test_dir,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )
    else:
        print('Error: no \'svhn\' or \'usps\' in the directory path')
        exit()
    
    testdl = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
    
    print('\nInferencing...\n')
    feature_extractor.to(device)
    label_predictor.to(device)
    imagenames, predictions = inference(feature_extractor, label_predictor, testdl, out_csv)
    print('\nOutputing...\n')
    out2csv(out_csv, imagenames, predictions)
    print('\nEnd\n')