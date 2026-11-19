import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.generator import generate_diffusion_images
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion
from helper_lib.utils import save_model

torch.manual_seed(42)

#check which GPU is available
device = (
  torch.device("mps")
  if torch.backends.mps.is_available()
  else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")

EPOCHS = 10

# DiffusionModel normalizes [0, 1] images with the CIFAR-10 mean/std itself
train_loader = get_data_loader('data', batch_size=64, train=True)

model = get_model("Diffusion")
optimizer = optim.AdamW(model.network.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.L1Loss()

model = train_diffusion(model, train_loader, optimizer, loss_fn, device=device, epochs=EPOCHS)

save_model(model, "models/diffusion_cifar10.pth")
generate_diffusion_images(model, num_samples=16, device=device, seed=0).save("diffusion_samples.png")
print("saved models/diffusion_cifar10.pth and diffusion_samples.png")