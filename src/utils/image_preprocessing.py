import torch, torchvision.transforms as T, cv2, numpy as np

def pre_process(inputs,noise_factor=0.1):
        """
        Function to add noise to photos in order to help model better generalize
        """ 
        
        inputs = T.ToTensor()(inputs)
        
        noisy = inputs+torch.randn_like(inputs) * noise_factor
        noisy = torch.clip(noisy,0.,1.)
        
        array = noisy.numpy()
        noise_img = (array.transpose((1,2,0)) * 255).astype(np.uint8)

        width, height = noise_img.shape[:2]
        aspect_ratio = width / height
        new_width = 256
        new_height = int(new_width / aspect_ratio)

        resized_img = cv2.resize(noise_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        return resized_img