from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from model import NSASpyware
from dataset import ImgDataset
import sys, os
from random import shuffle
from datetime import datetime
from helper import train_one_epoch, validate_one_epoch

TIMESTAMP = datetime.now().strftime('%Y%m%d')


def main(image_train_dataset, image_val_dataset):
    """Main function to train the model"""
    EPOCHS = 35
    WRITER = SummaryWriter()
    DEVICE = torch.device('mps')
    BEST_VAL_ACC = 0.0

    model = NSASpyware()
    model.to(DEVICE)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.002)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = image_train_dataset.get_dataloader()
    valid_loader = image_val_dataset.get_dataloader()

    for epoch in range(EPOCHS):

        # Train the model
        model.train(True)
        avg_tloss = train_one_epoch(
            train_loader, model, optimizer, loss_fn, DEVICE, epoch, EPOCHS)

        tb_x = epoch * len(train_loader)
        WRITER.add_scalar('Loss/train', avg_tloss, tb_x)

        # Validate the model
        model.train(False)
        avg_vloss, val_acc = validate_one_epoch(
            valid_loader, model, loss_fn, DEVICE)

        print(f'LOSS train {avg_tloss:.4f} valid {avg_vloss:.4f}')
        print(f'ACCURACY valid {val_acc:.4f}')

        WRITER.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_tloss, 'Validation': avg_vloss},
                           epoch + 1)
        WRITER.add_scalars('Validation Accuracy', {
                           'Accuracy': val_acc}, epoch + 1)
        WRITER.flush()

        # Track best performance, and save the model's state
        if val_acc > BEST_VAL_ACC:
            model_path = f'./saved_models/{epoch}_model_{int(val_acc * 100)}_{TIMESTAMP}.pt'
            torch.save(model.state_dict(), model_path)
            BEST_VAL_ACC = val_acc


if __name__ == '__main__':

    with os.scandir("../training_data/combined") as dir:
        images = [file.name for file in dir]
        shuffle(images)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5])
    ])

    train_images = images[:6200]
    val_images = images[6200:]

    image_train_dataset = ImgDataset(train_images, transform)
    image_val_dataset = ImgDataset(val_images, transform)

    main(image_train_dataset, image_val_dataset)
