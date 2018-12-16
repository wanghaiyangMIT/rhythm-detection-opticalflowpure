import os
import sys
import json
import cv2
from PIL import Image, ImageDraw
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter


def letterbox_image224(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w,h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas

def prep_image_to_tensor(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Returns a tensor
    """
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    orig_im = img #img is a nadrry
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image224(orig_im, (inp_dim, inp_dim)))
    img = img.astype(np.uint8)
    img_ = transform(img)
    #return img_, orig_im, dim
    return img_

class DataError(Exception):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

def collate_fn(batch):
    video_list = []
    label_list = []
    for video, label in batch:
        if len(label.shape) == 1:
            continue
        video_list += [video]
        label_list += [label]
    if len(label_list) < 1:
        zero = torch.tensor([0])
        return zero, zero
    video = torch.cat(video_list)
    label = torch.cat(label_list)
    return video, label

def is_peak(t, x):
    """
    t:  float
    x:  (T,)
    """
    if t == 0:
        return x[t] > x[t + 1]
    elif t == x.shape[0] - 1:
        return x[t] > x[t - 1]
    else:
        return x[t] > x[t + 1] and x[t] > x[t - 1]
        
