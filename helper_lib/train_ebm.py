import torch
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.generator import generate_ebm_images
from helper_lib.model import get_model
from helper_lib.trainer import train_ebm
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

# EBM works in [-1, 1], the same range the Langevin sampler clamps to
train_loader = get_data_loader('data', batch_size=128, train=True, normalize=True)

model = get_model("EnergyModel")
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.0, 0.999))

model = train_ebm(model, train_loader, optimizer, device=device, epochs=EPOCHS)

save_model(model, "models/ebm_cifar10.pth")
generate_ebm_images(model, num_samples=16, device=device, seed=0).save("ebm_samples.png")
print("saved models/ebm_cifar10.pth and ebm_samples.png")
