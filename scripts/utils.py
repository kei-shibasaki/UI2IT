import json
from collections import OrderedDict
from PIL import Image
import random 

import cv2
import numpy as np
import requests
import torch
from torch.nn import functional as F


def load_option(opt_path):
    with open(opt_path, 'r') as json_file:
        json_obj = json.load(json_file)
        return json_obj

def pad_tensor(x, divisible_by=8, mode='reflect'):
    if len(x.shape)==5:
        b,f,c,h,w = x.shape
        x = x.reshape(b*f,c,h,w)
    else:
        f = None
        _,_,h,w = x.shape
    
    nh = h//divisible_by+1 if h%divisible_by!=0 else h//divisible_by
    nw = w//divisible_by+1 if w%divisible_by!=0 else w//divisible_by
    nh, nw = int(nh*divisible_by), int(nw*divisible_by)
    pad_h, pad_w = nh-h, nw-w

    x = F.pad(x, [0,pad_w,0,pad_h], mode)

    if f is not None:
        x = x.reshape(b,f,c,nh,nw)

    return x

def load_img_uint8(img_path, to_tensor=False):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    if to_tensor:
        return torch.tensor(img, dtype=torch.uint8)
    else:
        return img

def read_img(path):
    # Return tensor with (C, H, W), RGB, [0, 1].
    img = cv2.imread(path).astype(np.float32) / 255.
    img = img2tensor(img, bgr2rgb=True, float32=True)

    return img

def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def tensor2ndarray(tensor):
    # Pytorch Tensor (B, C, H, W), [0, 1] -> ndarray (B, H, W, C) [0, 255]
    img = tensor.detach()
    img = img.cpu().permute(0,2,3,1).numpy()
    img = np.clip(img, a_min=0, a_max=1.0)
    img = (img*255).astype(np.uint8)
    return img

def send_line_notify(line_notify_token, nortification_message):
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'{nortification_message}'}
    requests.post(line_notify_api, headers=headers, data=data)

def arrange_images(images):
    # Input: list of PIL Image
    n = len(images)
    w, h = images[0].width, images[0].height
    out = Image.new('RGB', size=(n*w, h), color=0)
    for i, img in enumerate(images):
        out.paste(img, box=(i*w, 0))
    return out

