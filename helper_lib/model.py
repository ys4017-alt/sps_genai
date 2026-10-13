import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class AssignmentCNN(nn.Module):
    def __init__(self):
        super(AssignmentCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv2(x)))  # 32x32 -> 16x16
        x = x.view(-1, 32 * 16 * 16)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    

def get_model(model_name):
    # TODO (Module 4 Activity): add FCNN, CNN, EnhancedCNN
    if model_name == "AssignmentCNN":
        model = AssignmentCNN()
    else:
        raise ValueError(f"unknown model: {model_name}")

    return model
