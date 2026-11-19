import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
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
#A4
def swish(x):
    return x * torch.sigmoid(x)

#A4
class EnergyModel(nn.Module):
    def __init__(self):
        super(EnergyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)   # 32 -> 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 16 -> 8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 8 -> 4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # 4 -> 2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)

#A4
class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        # x: (B, 1, 1, 1) noise variance -> (B, 1, 1, 2 * num_frequencies)
        x = x.expand(-1, 1, 1, self.num_frequencies)
        return torch.cat([torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)], dim=-1)

#A4
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.proj(x)
        x = swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual

#A4
class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)

#A4
class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for block in self.blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x

#A4
class UNet(nn.Module):
    def __init__(self, num_channels=3, embedding_dim=32):
        super().__init__()
        self.num_channels = num_channels
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.embedding = SinusoidalEmbedding(num_frequencies=embedding_dim // 2)

        self.down1 = DownBlock(32, in_channels=32 + embedding_dim, block_depth=2)  # 32 -> 16
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)                  # 16 -> 8
        self.down3 = DownBlock(96, in_channels=64, block_depth=2)                  # 8 -> 4

        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)

        self.up1 = UpBlock(96, in_channels=128, block_depth=2)  # 4 -> 8
        self.up2 = UpBlock(64, in_channels=96, block_depth=2)   # 8 -> 16
        self.up3 = UpBlock(32, in_channels=64, block_depth=2)   # 16 -> 32

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances).permute(0, 3, 1, 2)  # (B, 32, 1, 1)
        noise_emb = F.interpolate(noise_emb, size=noisy_images.shape[-2:], mode="nearest")
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        return self.final(x)

#A4
def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)
    return noise_rates, signal_rates

#A4
class DiffusionModel(nn.Module):
    def __init__(self, network, schedule_fn=offset_cosine_diffusion_schedule, ema_decay=0.999):
        super().__init__()
        self.network = network
        self.ema_network = copy.deepcopy(network).requires_grad_(False)
        self.schedule_fn = schedule_fn
        self.ema_decay = ema_decay
        # CIFAR-10 per-channel stats
        self.register_buffer("normalizer_mean", torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer("normalizer_std", torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        network = self.network if training else self.ema_network
        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)
        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.lerp_(param, 1.0 - self.ema_decay)
        return loss.item()

    @torch.no_grad()
    def generate(self, initial_noise, diffusion_steps=20):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            t = torch.full((initial_noise.size(0), 1, 1, 1), 1.0 - step * step_size, device=initial_noise.device)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)
            next_noise_rates, next_signal_rates = self.schedule_fn(t - step_size)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return torch.clamp(pred_images * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

def get_model(model_name):
    # TODO (Module 4 Activity): add FCNN, CNN, EnhancedCNN
    if model_name == "AssignmentCNN":
        model = AssignmentCNN()
    #A3 models: Generator, Discriminator
    elif model_name == "Generator":
        model = Generator()
    elif model_name == "Discriminator":
        model = Discriminator()
    elif model_name == "EnergyModel":
        model = EnergyModel()
    elif model_name == "Diffusion":
        model = DiffusionModel(UNet(num_channels=3))
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

    x = torch.rand(8, 3, 32, 32) * 2 - 1
    energy = EnergyModel()(x)
    assert energy.shape == (8, 1), energy.shape
    print("EnergyModel:", tuple(x.shape), "->", tuple(energy.shape))

    diffusion = get_model("Diffusion")
    pred_noise = diffusion.network(x, torch.rand(8, 1, 1, 1))
    assert pred_noise.shape == x.shape, pred_noise.shape
    samples = diffusion.generate(torch.randn(8, 3, 32, 32), diffusion_steps=5)
    assert samples.shape == (8, 3, 32, 32) and samples.min() >= 0 and samples.max() <= 1
    print("UNet:", tuple(x.shape), "->", tuple(pred_noise.shape), "| generate ->", tuple(samples.shape))