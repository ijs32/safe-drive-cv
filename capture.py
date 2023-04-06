import numpy as np
import cv2, torch, sys
import utils.image_preprocessing as ip

def main(type):
    vidcap = cv2.VideoCapture(0)
    _, image = vidcap.read()
    for i in range(1000):

        noise_tensor = ip.add_noise(image)
        array = noise_tensor.numpy()
        noise_img = (array.transpose((1,2,0)) * 255).astype(np.uint8)

        width, height = noise_img.shape[:2]
        aspect_ratio = width / height
        new_width = 256
        new_height = int(new_width / aspect_ratio)

        resized_img = cv2.resize(noise_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        cv2.imshow('frame',resized_img)
        cv2.imwrite(f"./training_data/combined/{type}_frame{i}.jpg", resized_img) # save the img 

        _, image = vidcap.read()
        print('Read a new frame: ', i)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python capture.py (snapchat or imessage)")
        sys.exit(1)
    main(sys.argv[1])