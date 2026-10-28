import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.utils import save_model
from helper_lib.generator import generate_gan_images


torch.manual_seed(42)

#check which GPU is available
device = (
  torch.device("mps")
  if torch.backends.mps.is_available()
  else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")

train_loader = get_data_loader('data', batch_size=64, train=True, dataset="MNIST")

generator = get_model("Generator")
discriminator = get_model("Discriminator")
criterion = nn.BCELoss()

g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas =(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

generator, discriminator = train_gan(
  generator, discriminator, train_loader, criterion, g_optimizer, d_optimizer, device=device, epochs=10
)

save_model(generator, "models/gan_mnist_generator.pth")
generate_gan_images(generator, num_samples=64, device=device, seed=0).save("gan_samples.png")
print("save gan_minist_generator.pth and gan_samples.png")

