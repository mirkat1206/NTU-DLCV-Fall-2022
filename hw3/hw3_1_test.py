import os
import sys
import clip
import glob
import json
import torch
import numpy as np
from PIL import Image


# Utils
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Inference
def inference(image_paths, id2label, out_csv):
    model, preprocess = clip.load('ViT-B/32', device)
    # 0
    # text_inputs = torch.cat([clip.tokenize(f"This is a photo of {label}") for _, label in id2label.items()]).to(device)
    # 1
    text_inputs = torch.cat([clip.tokenize(f"This is a {label} image") for _, label in id2label.items()]).to(device)
    # 2
    # text_inputs = torch.cat([clip.tokenize(f"No {label}, no score") for _, label in id2label.items()]).to(device)

    imagenames = []
    predictions = []
    with torch.no_grad():
        for i in range(len(image_paths)):
            print(i)
            image = Image.open(image_paths[i])
            image_input = preprocess(image).unsqueeze(0).to(device)

            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pred_values, pred_indices = similarity[0].topk(1)

            for value, index in zip(pred_values, pred_indices):
                index = str(int(index))
                imagenames.append(os.path.basename(image_paths[i]))
                predictions.append(index)
                # print(f"{id2label[index]:>4s}: ({index}) {100 * value.item():.2f}%")
    out2csv(out_csv, imagenames, predictions)

# Output
def out2csv(out_csv, imagenames, predictions):
    out = np.array([imagenames, predictions])
    out = np.rot90(out)
    out = out[out[:, 0].argsort()]
    out = np.insert(out, 0, ['filename', 'label'], 0)
    np.savetxt(out_csv, out, delimiter=',', fmt='%s')

# Sample 3 images and visualize the probability of the top-5 similarity scores
def top5prob(id2label):
    model, preprocess = clip.load('ViT-B/32', device)
    # 0
    # text_inputs = torch.cat([clip.tokenize(f"This is a photo of {label}") for _, label in id2label.items()]).to(device)
    # 1
    # text_inputs = torch.cat([clip.tokenize(f"This is a {label} image") for _, label in id2label.items()]).to(device)
    # 2
    text_inputs = torch.cat([clip.tokenize(f"No {label}, no score") for _, label in id2label.items()]).to(device)

    image_paths = [
        './hw3_data/p1_data/val/0_450.png',
        './hw3_data/p1_data/val/2_493.png',
        './hw3_data/p1_data/val/48_484.png',
    ]
    with torch.no_grad():
        for i in range(len(image_paths)):
            fp = image_paths[i]
            fn = os.path.basename(fp)
            golden_label = fn[:fn.find('_')]

            image = Image.open(fp)
            image_input = preprocess(image).unsqueeze(0).to(device)

            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pred_values, pred_indices = similarity[0].topk(5)

            print("\nImage {:d}: {}".format(i + 1, id2label[golden_label]))
            for value, index in zip(pred_values, pred_indices):
                index = str(int(index))
                print(f"{id2label[index]:8s}: \t({index}) \t{100 * value.item():.2f}%")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Error: wrong format')
        print('\tpython3 hw2_3_test.py <test image dirpath> <id2label.json path> <output csv filepath>')
        exit()
    
    test_dir = sys.argv[1]
    out_csv = sys.argv[3]

    with open(sys.argv[2]) as f:
        id2label = json.load(f)

    image_paths = []
    for fp in glob.glob(test_dir + '/*.png'):
        image_paths.append(fp)
    
    # top5prob(id2label)

    print('\nInferencing...\n')
    inference(image_paths, id2label, out_csv)
    print('\nEnd\n')
