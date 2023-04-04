import torch
import torch.nn as nn

class NSASpyware(nn.Module):
    """
    ### NLP Model
    """

    def __init__(self):

        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 16, 7),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.20),
            
            nn.Conv2d(16, 16, 7),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.20),
            
            nn.Conv2d(16, 16, 7),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(2),
            nn.Dropout(0.20),
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(16, 1)
        )

    def forward(self, img):
        x = self.conv_stack(img)
        x = x.flatten(start_dim=1)
        x = self.linear_stack(x)
        x = torch.sigmoid(x.view(-1, 1))
        return x.squeeze()
