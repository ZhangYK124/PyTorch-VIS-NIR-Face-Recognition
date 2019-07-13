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

class Dataset(data.Dataset):
    def __init__(self):
        
        self.vis_root = '/media/hyo/文档/VIS-NIR/CASIA_VIS_NIR/VIS_Aligned/'
        self.nir_root = '/media/hyo/文档/VIS-NIR/CASIA_VIS_NIR/NIR_Aligned/'
        self.vis_img_list = os.listdir(self.vis_root)
        self.nir_img_list = os.listdir(self.nir_root)

        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        
    def __len__(self):
        return min(len(self.vis_img_list),len(self.nir_img_list))
    
    def __getitem__(self, index):
        vis_img = Image.open(os.path.join(self.vis_root,self.vis_img_list[index],self.vis_img_list[index]+'.jpg'))
        
        # nir_name = choice(self.nir_img_list)
        nir_name = self.nir_img_list[index]
        nir_img = Image.open(os.path.join(self.nir_root,nir_name,nir_name+'.jpg'))
        nir_img = nir_img.convert("RGB")
        
        vis_left_eye_img = Image.open(os.path.join(self.vis_root,self.vis_img_list[index],'left_eye.jpg'))
        vis_right_eye_img = Image.open(os.path.join(self.vis_root,self.vis_img_list[index],'right_eye.jpg'))
        
        nir_left_eye_img = Image.open(os.path.join(self.nir_root,nir_name,'left_eye.jpg'))
        nir_right_eye_img = Image.open(os.path.join(self.nir_root,nir_name,'right_eye.jpg'))
        
        seed = random.random()
        if seed>0.5:
            temp = vis_left_eye_img
            vis_left_eye_img = vis_right_eye_img
            vis_right_eye_img = temp
            
            temp = nir_left_eye_img
            nir_left_eye_img = nir_right_eye_img
            nir_right_eye_img = temp
            
            vis_left_eye_img = vis_left_eye_img.transpose(Image.FLIP_LEFT_RIGHT)
            vis_right_eye_img = vis_right_eye_img.transpose(Image.FLIP_LEFT_RIGHT)
            nir_left_eye_img = nir_left_eye_img.transpose(Image.FLIP_LEFT_RIGHT)
            nir_right_eye_img = nir_right_eye_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            vis_img = vis_img.transpose(Image.FLIP_LEFT_RIGHT)
            nir_img = nir_img.transpose(Image.FLIP_LEFT_RIGHT)
      
        batch = {}
        batch['vis'] = vis_img
        batch['nir'] = nir_img
        batch['vis_left_eye'] = vis_left_eye_img
        batch['vis_right_eye'] = vis_right_eye_img
        batch['nir_left_eye'] = nir_left_eye_img
        batch['nir_right_eye'] = nir_right_eye_img
        for k in batch:
            batch[k] = self.tf(batch[k])
        return batch
        