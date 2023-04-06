import torch, torchvision.transforms as T

def add_noise(inputs,noise_factor=0.1):
        """
        Function to add noise to photos in order to help model better generalize
        """ 
        transforms = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(60)
        ])
        inputs = transforms(inputs)
        
        noisy = inputs+torch.randn_like(inputs) * noise_factor
        noisy = torch.clip(noisy,0.,1.)
        return noisy