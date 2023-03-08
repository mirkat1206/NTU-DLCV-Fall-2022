import os
import clip
import json
import torch
from PIL import Image

class CLIPScore:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def __call__(self, predictions, images_root):
        """
        Input:
            predictions: dict of str
            images_root: str
        Return:
            clip_score: float
        """
        total_score = 0.

        for img_name, pred_caption in predictions.items():
            image_path = os.path.join(images_root, f"{img_name}.jpg")
            image = Image.open(image_path).convert("RGB")

            total_score += self.getCLIPScore(image, pred_caption)
        return total_score / len(predictions)

    def getCLIPScore(self, image, caption):
        """
        This function computes CLIPScore based on the pseudocode in the slides.
        Input:
            image: PIL.Image
            caption: str
        Return:
            cilp_score: float
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([caption]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
        
        cos_sim = torch.nn.functional.cosine_similarity(image_features, text_features).item()
        return 2.5 * max(cos_sim, 0)

if __name__ == "__main__":
    clip_score = CLIPScore()
    f = open('./out_clip/out16.json', 'r')
    pred = json.load(f)
    f.close()
    dirpath = './hw3_data/p2_data/images/val/'
    min_score = 2
    max_score = -1
    for image_name, caption in pred.items():
        img = Image.open(dirpath + image_name + '.jpg')
        score = clip_score.getCLIPScore(img, caption)
        if score < min_score:
            min_image_name = image_name
            min_score = score
        if score > max_score:
            max_image_name = image_name
            max_score = score
    print(min_image_name, min_score)
    print(max_image_name, max_score)