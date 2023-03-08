import csv 
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Utils
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device used: ", device)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

label2class = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']
class2label = {'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 'Calendar': 8, 'Candles': 9, 'Chair': 10, 'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15, 'Drill': 16, 'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 'Glasses': 25, 'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44, 'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 'Screwdriver': 51, 'Shelf': 52, 'Sink': 53, 'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57, 'TV': 58, 'Table': 59, 'Telephone': 60, 'ToothBrush': 61, 'Toys': 62, 'Trash_Can': 63, 'Webcam': 64}

# Custom Dataset & DataLoader
class HW4P2ATest(Dataset):
    def __init__(self, dirpath, csvpath, transform=None):
        self.dirpath = dirpath
        self.filenames = []
        with open(csvpath) as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                id, fn, klass = row
                if id != 'id':
                    self.filenames.append(fn)
        self.transform = transform
    
    def __getitem__(self, index):
        fn = self.filenames[index]
        image = Image.open(self.dirpath + '/' + fn)
        if self.transform is not None:
            image = self.transform(image)
        return image, fn
    
    def __len__(self):
        return len(self.filenames)



# dataloader = {
#     'train': DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4),
#     'val': DataLoader(valset, batch_size=64, shuffle=False, num_workers=4),
# }
# dataiter = iter(dataloader['train'])
# images, labels = dataiter.next()
# print('Image tensor in each batch:', images.shape, images.dtype)
# print('Image tensor in each batch:', labels.shape, labels.dtype)

# Model
class MyNN(nn.Module):
    def __init__(self, backbone_checkpoint=None, classifier_checkpoint=None, fix_backbone=False):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        if backbone_checkpoint is not None:
            self.backbone.load_state_dict(torch.load(backbone_checkpoint))
        for param in self.backbone.parameters():
            if fix_backbone:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 65)
        )
        if classifier_checkpoint is not None:
            self.classifier.load_state_dict(torch.load(classifier_checkpoint))
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def get_params_to_update(self):
        params_to_update = []
        for param in self.backbone.parameters():
            if param.requires_grad:
                params_to_update.append(param)
            else:
                break
        for param in self.classifier.parameters():
            if param.requires_grad:
                params_to_update.append(param)
            else:
                break
        return params_to_update

# Output
def out2csv(outpath, imagenames, predictions, in_csv):
    fout = open(outpath, "w")
    with open(in_csv) as fin:
        csvreader = csv.reader(fin)
        for row in csvreader:
            id, fn, klass = row
            if id != 'id':
                index = imagenames.index(fn)
                klass = predictions[index]
                fout.write(id + "," + fn + "," + klass + "\n")
            else:
                fout.write(id + "," + fn + "," + klass + "\n")
    fout.close()

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Error: wrong format')
        print('\tpython3 hw4_2_test.py <input csv filepath> <input image filepath> <output csv file> <A/B/C/D/E>')
        exit()

    in_csv = sys.argv[1]
    in_dir = sys.argv[2]
    out_csv = sys.argv[3]
    mode = sys.argv[4]

    # dataset & dataloader
    valset = HW4P2ATest(
        dirpath=in_dir,
        csvpath=in_csv,
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    )
    dataloader = {
        'val': DataLoader(valset, batch_size=64, shuffle=False, num_workers=4),
    }

    # model
    backbone_checkpoint="./bestcheckpoin/hw4_2C_backbone.pt"
    classifier_checkpoint="./bestcheckpoin/hw4_2C_classifier.pt"
    # if mode == 'A':
    #     backbone_checkpoint="./checkpoint/hw4_2A_backbone.pt"
    #     classifier_checkpoint="./checkpoint/hw4_2A_classifier.pt"
    # elif mode == 'B':
    #     backbone_checkpoint="./checkpoint/hw4_2B_backbone.pt"
    #     classifier_checkpoint="./checkpoint/hw4_2B_classifier.pt"
    # elif mode == 'C':
    #     backbone_checkpoint="./checkpoint/hw4_2C_backbone.pt"
    #     classifier_checkpoint="./checkpoint/hw4_2C_classifier.pt"
    # elif mode == 'D':
    #     backbone_checkpoint="./checkpoint/pretrain_model_SL.pt"
    #     classifier_checkpoint="./checkpoint/hw4_2D_classifier.pt"
    # else:
    #     backbone_checkpoint="./checkpoint/mybackbone.pt"
    #     classifier_checkpoint="./checkpoint/hw4_2E_classifier.pt"

    mynn = MyNN(
        backbone_checkpoint=backbone_checkpoint,
        classifier_checkpoint=classifier_checkpoint,
        fix_backbone=False
    )
    mynn = mynn.to(device)

    # inference
    mynn.eval()
    filenames = []
    klasses = []
    with torch.no_grad():
        for _, (images, fns) in enumerate(dataloader['val']):
            images = images.to(device) 

            outs = mynn(images)
            preds = outs.max(1, keepdim=True)[1]
            filenames.extend(fns)
            for pred in preds:
                klasses.append(label2class[pred])

    # output
    out2csv(out_csv, filenames, klasses, in_csv)

