import os, sys, torch.nn as nn, torch.optim as optim, torchvision.transforms as T
from classification_model import NSASpyware
from dataset import ImgDataset
from random import shuffle
from datetime import datetime
from trainer import Trainer


def main():
    """Main function to train the model"""

    with os.scandir("../training_data/classification_training_data/") as dir:
        images = [file.name for file in dir]
        shuffle(images)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5])
    ])

    train_images = images[:3800]
    val_images = images[3800:]

    image_train_dataset = ImgDataset(train_images, transform)
    image_val_dataset = ImgDataset(val_images, transform)
    
    timestamp = datetime.now().strftime('%Y%m%d')
    epochs = 35
    seed = 42

    model = NSASpyware()
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.002)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = image_train_dataset.get_dataloader()
    valid_loader = image_val_dataset.get_dataloader()

    # Train the model.
    model_trainer = Trainer(model, loss_fn, optimizer)
    model_trainer.set_loaders(train_loader, valid_loader)
    model_trainer.set_tensorboard(f"pytorch_{timestamp}")
    model_trainer.train(epochs, seed)

if __name__ == '__main__':

    main()
