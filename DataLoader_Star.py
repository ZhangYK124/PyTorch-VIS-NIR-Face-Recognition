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

import pdb

class Dataset(data.Dataset):
    def __init__(self):
        self.vis_root = '/media/hyo/文档/VIS-NIR/CASIA_VIS_NIR/VIS_Aligned/'
        self.nir_root = '/media/hyo/文档/VIS-NIR/CASIA_VIS_NIR/NIR_Aligned/'
        self.sketch_root = '/media/hyo/文档/VIS-NIR/SKETCH/cropped_sketch'

        self.vis_img_list = os.listdir(self.vis_root)
        self.nir_img_list = os.listdir(self.nir_root)
        self.sketch_img_list = os.listdir(self.sketch_root)

        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return min(len(self.vis_img_list),len(self.nir_img_list),len(self.sketch_img_list))
    
    def __getitem__(self, index):
        nir_img = Image.open(os.path.join(self.nir_root, self.nir_img_list[index], self.nir_img_list[index] + '.jpg'))
        nir_img = nir_img.convert("RGB")
        
        # vis_name = choice(self.vis_img_list)
        vis_name = self.vis_img_list[index]
        vis_img = Image.open(os.path.join(self.vis_root, vis_name, vis_name + '.jpg'))
        
        # sketch_name = choice(self.sketch_img_list)
        sketch_name = self.sketch_img_list[index]
        sketch_img = Image.open(os.path.join(self.sketch_root,sketch_name))
        
        seed = random.random()
        if seed >0.5:
            vis_img = vis_img.transpose(Image.FLIP_LEFT_RIGHT)
            nir_img = nir_img.transpose(Image.FLIP_LEFT_RIGHT)
            sketch_img = sketch_img.transpose(Image.FLIP_LEFT_RIGHT)
            
        vis_img = vis_img.rotate(random.uniform(-1,1))
        nir_img = nir_img.rotate(random.uniform(-1,1))
        sketch_img = sketch_img.rotate(random.uniform(-1,1))
        
        batch = {}
        batch['vis'] = vis_img
        batch['nir'] = nir_img
        batch['sketch'] = sketch_img
        
        for k in batch:
            batch[k] = self.tf(batch[k])
            
        return batch
        
        
        
    

