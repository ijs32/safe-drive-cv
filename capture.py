import numpy as np
import cv2
import torchvision.transforms as T
import torch 
from PIL import Image

def add_noise(inputs,noise_factor=0.3):
     """
     Function to add noise to photos in order to help model better generalize
     """
     noisy = inputs+torch.randn_like(inputs) * noise_factor
     noisy = torch.clip(noisy,0.,1.)
     return noisy


vidcap = cv2.VideoCapture(0)
success,image = vidcap.read()
count = 0
while success:

    noise_tensor = add_noise(T.ToTensor()(image))
    array = noise_tensor.numpy()
    noise_img = (array.transpose((1,2,0)) * 255).astype(np.uint8)

    cv2.imwrite(f"./training_data/spotify/frame{count}.jpg", noise_img) # save the img 

    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1