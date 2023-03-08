import csv 
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Utils
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
print("Device used: ", device)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

label2class = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']
class2label = {'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 'Calendar': 8, 'Candles': 9, 'Chair': 10, 'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15, 'Drill': 16, 'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 'Glasses': 25, 'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44, 'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 'Screwdriver': 51, 'Shelf': 52, 'Sink': 53, 'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57, 'TV': 58, 'Table': 59, 'Telephone': 60, 'ToothBrush': 61, 'Toys': 62, 'Trash_Can': 63, 'Webcam': 64}

# Custom Dataset & DataLoader
class HW4P2A(Dataset):
    def __init__(self, dirpath, csvpath, transform=None):
        self.dirpath = dirpath
        self.filenames = []
        self.labels = []
        with open(csvpath) as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                id, fn, klass = row
                if id != 'id':
                    self.filenames.append(fn)
                    self.labels.append(class2label[klass])
        self.transform = transform
    
    def __getitem__(self, index):
        fn = self.filenames[index]
        image = Image.open(self.dirpath + '/' + fn)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]
    
    def __len__(self):
        return len(self.filenames)

trainset = HW4P2A(
    dirpath='./hw4_data/office/train/',
    csvpath='./hw4_data/office/train.csv',
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop((128)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30), expand=False),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
)
valset = HW4P2A(
    dirpath='./hw4_data/office/val/',
    csvpath='./hw4_data/office/val.csv',
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
)
print("# of images in trainset: ", len(trainset))
print("# of images in valset: ", len(valset)) 

dataloader = {
    'train': DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4),
    'val': DataLoader(valset, batch_size=64, shuffle=False, num_workers=4),
}
dataiter = iter(dataloader['train'])
images, labels = dataiter.next()
print('Image tensor in each batch:', images.shape, images.dtype)
print('Image tensor in each batch:', labels.shape, labels.dtype)

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

mynn = MyNN(
    backbone_checkpoint="./hw4_data/pretrain_model_SL.pt",
    # classifier_checkpoint="./checkpoint/hw4_2A_classifier.pt",
    fix_backbone=True
)
mynn = mynn.to(device)
optimizer = torch.optim.Adam(mynn.get_params_to_update())
# optimizer = torch.optim.Adam(mynn.get_params_to_update(), lr=0.02, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss()

# Train
def train(epoch, init_epoch):
    mynn.train()
    running_loss = 0
    correct = 0
    for _, (images, labels) in enumerate(dataloader['train']):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outs = mynn(images)
        preds = outs.max(1, keepdim=True)[1]
        loss = criterion(outs, labels)
        running_loss += loss.item()
        correct += preds.eq(labels.view_as(preds)).sum().item()
        loss.backward()
        optimizer.step()

    running_loss /= len(dataloader['train'].dataset)
    accuracy = 100. * correct / len(dataloader['train'].dataset)
    print('Train Epoch: {:3d}\nAverage Loss: {:.6f}\tAccuracy: {}/{} ({:.4f}%)'.format(
        epoch + init_epoch,
        running_loss,
        correct,
        len(dataloader['train'].dataset),
        accuracy,
    ))
    return accuracy

def test():
    mynn.eval()
    running_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (images, labels) in enumerate(dataloader['val']):
            images, labels = images.to(device), labels.to(device)

            outs = mynn(images)
            preds = outs.max(1, keepdim=True)[1]
            running_loss = criterion(outs, labels).item()
            correct += preds.eq(labels.view_as(preds)).sum().item()

        running_loss /= len(dataloader['val'].dataset)
        accuracy = 100. * correct / len(dataloader['val'].dataset)
        print('Test Average Loss: {:.6f}\tAccuracy: {}/{} ({:.4f}%)'.format(
            running_loss,
            correct,
            len(dataloader['val'].dataset),
            accuracy,
        ))
    return accuracy

best_accuracy = -1
for epoch in range(200):
    accuracy = train(epoch, 0)
    accuracy = test()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # torch.save(mynn.backbone.state_dict(), './checkpoint/hw4_2D_backbone.pt')
        torch.save(mynn.classifier.state_dict(), './checkpoint/hw4_2D_classifier.pt')
