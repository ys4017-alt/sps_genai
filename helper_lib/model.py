import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

#A2
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

#A3
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        self.fc = nn.Linear(z_dim, 7*7*128)
        self.reshape = lambda x: x.view(x.size(0), 128, 7,7)
        
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(True)
        
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.fc(x)
        x = self.reshape(x) #(B, 100) to (b, 128,7,7)
        
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.deconv2(x)
        x = self.tanh(x)
        
        return x

#A3
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 *7, 1 )
        self.sigmoid = nn.Sigmoid()        

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x

def get_model(model_name):
    # TODO (Module 4 Activity): add FCNN, CNN, EnhancedCNN
    if model_name == "AssignmentCNN":
        model = AssignmentCNN()
    #A3 models: Generator, Discriminator
    elif model_name == "Generator":
        model = Generator()
    elif model_name == "Discriminator":
        model = Discriminator()
    else:
        raise ValueError(f"unknown model: {model_name}")

    return model

if __name__ == "__main__":
    z = torch.randn(8, 100)
    fake = Generator()(z)
    assert fake.shape == (8, 1, 28, 28), fake.shape
    prob = Discriminator()(fake)
    assert prob.shape == (8,1), prob.shape
    assert ((prob>=0) & (prob<=1)).all()
    print("Generator:", tuple(z.shape), "->", tuple(fake.shape))
    print("Discriminator:", tuple(fake.shape), "->", tuple(prob.shape))
