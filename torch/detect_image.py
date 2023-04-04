
import torch
from model import NSASpyware
import numpy as np
import cv2, torch
import torchvision.transforms as T

def test(resized_img):
    """Main function to train the model"""
    DEVICE = torch.device('mps')

    model = NSASpyware()
    model.load_state_dict(torch.load(
        './saved_models/2_model_9327_20230403.pt'))
    model.eval()
    model.to(DEVICE)

    resized_img = resized_img.to(DEVICE).float()
    output = model(resized_img)
    print(output)


def add_noise(inputs,noise_factor=0.3):
     """
     Function to add noise to photos in order to help model better generalize
     """
     noisy = inputs+torch.randn_like(inputs) * noise_factor
     noisy = torch.clip(noisy,0.,1.)
     return noisy


vidcap = cv2.VideoCapture(0)
success,image = vidcap.read()
while success:

    noise_tensor = add_noise(T.ToTensor()(image))
    array = noise_tensor.numpy()
    noise_img = (array.transpose((1,2,0)) * 255).astype(np.uint8)

    width, height = noise_img.shape[:2]
    aspect_ratio = width / height
    new_width = 256
    new_height = int(new_width / aspect_ratio)

    resized_img = cv2.resize(noise_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    cv2.imshow('frame',resized_img)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5])
    ])
    normalized_tensor = transform(resized_img)

    test(normalized_tensor.unsqueeze(0))

    success,image = vidcap.read()





