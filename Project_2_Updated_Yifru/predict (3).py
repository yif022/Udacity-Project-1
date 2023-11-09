import argparse
import torch
import json
from torchvision import models
from functions import load_checkpoint
from utility_func import predict


parser = argparse.ArgumentParser(description = 'Predict flower image') 

parser.add_argument('image_path', type = str, default = 'flowers/test/34/image_06961.jpg',  help = 'path to image')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'path to checkpoint')
parser.add_argument('--arch', type = str, default = 'densenet161', help = 'choose densenet161 or densenet121 CNN architecture')
parser.add_argument('--top_k', type = int, default = 5, help = 'number of top class to be printed')
parser.add_argument('--gpu', type = bool, default = False, help = 'use gpu')
parser.add_argument('--cat_to_name', type = str, default = 'cat_to_name.json', help = 'display class names')

args = parser.parse_args()

image_path = args.image_path
save_dir = args.save_dir
topk = args.top_k
arch = args.arch
gpu = args.gpu
cat_names = args.cat_to_name

def main():
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f, strict=False)

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    model = models.__dict__[arch](pretrained=True)
    model.to(device)
    load_model = load_checkpoint(model, save_dir)

    top_probs, top_labels, top_flowers = predict(image_path, model, cat_to_name, topk, device)
    print(f"The probability of most likely flower class is: {top_probs}")
    print(f"The most likely flower class is: {top_labels}")
    print(f"The names of the most likely flower class is: {top_flowers}")

if __name__ =="__main__":
    main()