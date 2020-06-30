import os, sys, pdb, time
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CarDataset(Dataset):
    def __init__(self, roots, W=None, H=None, split="train"):
        super(CarDataset, self).__init__()
        self.all_files = []
        self.W = W
        self.H = H
        # make a list of all file
        for root in roots:
            for f in os.listdir(root):
                if ".png" in f or ".jpg" in f:
                    self.all_files.append(os.path.join(root, f))
        self.all_files.sort()
        # if it is train set use the first 90% of the dataset
        if split == "train":
            self.all_files = self.all_files[0:int(len(self.all_files)*0.9)]
        elif split == "test":
            self.all_files = self.all_files[int(len(self.all_files)*0.9):]
        self.transform_image = transforms.ToTensor()
    
    def __len__(self):
        return len(self.all_files)
    
    def make_label(self, fname):
        bname = os.path.basename(fname)
        tokens = bname.split("_")
        angle = float(tokens[1])
        throttle = float(tokens[2].replace(".png",""))
        return angle, throttle


    def __getitem__(self, idx):
        angle, throttle = self.make_label(self.all_files[idx])
        sample = {  "image"      : Image.open(self.all_files[idx]),
                    "throttle"   : torch.tensor(throttle).float(),
                    "steer"      : torch.tensor(angle).float(),
                    "path"       : self.all_files[idx]}
        sample["image"] = self.transform_image(sample["image"])
        return sample