import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

import glob
import os
import random
from random import choice
import matplotlib.pyplot as plt

from PIL import Image,ImageEnhance

from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face

import pdb

class DataLoader(data.Dataset):
    def __init__(self):
        
        self.vis_root = '/media/hyo/文档/VIS-NIR/CASIA_VIS_NIR/NIR_Aligned/'
        self.nir_root = '/media/hyo/文档/VIS-NIR/CASIA_VIS_NIR/VIS_Aligned/'
        self.vis_img_list = glob.glob(self.vis_root)
        self.nir_img_list = glob.glob(self.nir_root)
        
    def __len__(self):
        return min(len(glob.glob(self.vis_root)),len(glob.glob(self.nir_root)))
    
    def __getitem__(self, index):
        nir_img = Image.open(self.nir_img_list[index])
        nir_img = nir_img.convert("RGB")
        vis_img = Image.open(choice(self.vis_img_list))
        _ , nir_landmark = detect_faces(nir_img)
        _ , vis_landmark = detect_faces(vis_img)
        