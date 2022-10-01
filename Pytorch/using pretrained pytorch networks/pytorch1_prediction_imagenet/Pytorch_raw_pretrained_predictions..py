# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:09:32 2022

@author: DELL
"""

from torchvision import models

import torch

print(dir(models))



alexnet = models.alexnet(pretrained=True)
# You will see a similar output as below
# Downloading: "https://download.pytorch.org/models/alexnet-owt- 4df8aa71.pth" to /home/hp/.cache/torch/checkpoints/alexnet-owt-4df8aa71.pth
print(alexnet)


from torchvision import transforms

transform = transforms.Compose([            #[1]
transforms.Resize(256),                    #[2]

transforms.CenterCrop(224),                #[3]

transforms.ToTensor(),                     #[4]

transforms.Normalize(                      #[5]

mean=[0.485, 0.456, 0.406],                #[6]

std=[0.229, 0.224, 0.225]                  #[7]

 )])


# Import Pillow

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = Image.open("goldFish.jfif")
plt.imshow(np.array(img))



img_t = transform(img)

batch_t = torch.unsqueeze(img_t, 1)

alexnet.eval()


out = alexnet(batch_t)

print(out.shape)

with open('imagenet_classes.txt') as f:

  labels = [line.strip() for line in f.readlines()]
  
  
  
_, index = torch.max(out, 1)  ## one = dimension

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

print(labels[index[0]], percentage[index[0]].item())

