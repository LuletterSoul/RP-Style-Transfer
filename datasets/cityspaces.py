
from sampler import InfiniteSamplerWrapper
import network as net
import argparse
from datetime import time
from logging import Logger
from pathlib import Path
import time
import os
import cv2

import torch
# import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
import yaml
from tqdm import tqdm
import logging
import torchvision
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CityspacesDataset(data.Dataset):
    def __init__(self, img_dir,transform, img_size=256):
        super(CityspacesDataset, self).__init__()
        self.img_dir = img_dir
        self.img_names = os.listdir(self.img_dir)
        self.transform = transform
        self.img_size = img_size

        ignore_label = -1

        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}

    def __getitem__(self, index):
        img_path = os.path.join(
            self.img_dir, self.img_names[index])
        # img = Image.open(img_path).convert('RGB')
    
        img = cv2.imread(img_path)

        content = img[:,:self.img_size,:] 
        label = img[:,self.img_size: self.img_size * 2, :]

        content = Image.fromarray(cv2.cvtColor(content, cv2.COLOR_BGR2RGB))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = self.convert_label(label)
        label = Image.fromarray(label)

        content = self.transform(content)

        return content, label
    
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __len__(self):
        return len(self.img_names)

    def name(self):
        return 'CityspacesDataset'