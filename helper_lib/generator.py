import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

def generate_samples(model, device, num_samples=10):
    # TODO: generate num_samples points in the latent space, run the decoder to construct an image, and plot the samples on a grid
    plt.show()
    
def generate_gan_images(generator, num_samples=16, device='cpu', seed=None):
    rng = torch.Generator().manual_seed(seed) if seed is not None else None
    z = torch.randn(num_samples, generator.z_dim, generator=rng).to(device)
    generator.eval()
    with torch.no_grad():
        images = generator(z)
    images = (images +1)/2
    grid = make_grid(images.cpu(), nrow=8, padding=2)
    return to_pil_image(grid)

#A4
def langevin_sample(energy_model, images, steps=60, step_size=10.0, noise_std=0.005):
    energy_model.eval()
    for _ in range(steps):
        with torch.no_grad():
            images = (images + torch.randn_like(images) * noise_std).clamp(-1.0, 1.0)
        images.requires_grad_(True)
        energy = energy_model(images)
        grads, = torch.autograd.grad(energy, images, grad_outputs=torch.ones_like(energy))
        with torch.no_grad():
            images = (images - step_size * grads.clamp(-0.03, 0.03)).clamp(-1.0, 1.0)
    return images.detach()

#A4
def generate_ebm_images(energy_model, num_samples=16, steps=256, device='cpu', seed=None):
    with torch.random.fork_rng():
        if seed is not None:
            torch.manual_seed(seed)
        start = (torch.rand(num_samples, 3, 32, 32) * 2 - 1).to(device)
        images = langevin_sample(energy_model, start, steps=steps, step_size=10.0, noise_std=0.01)
    images = (images + 1) / 2
    grid = make_grid(images.cpu(), nrow=8, padding=2)
    return to_pil_image(grid)

#A4
def generate_diffusion_images(diffusion_model, num_samples=16, diffusion_steps=20, device='cpu', seed=None):
    rng = torch.Generator().manual_seed(seed) if seed is not None else None
    noise = torch.randn(num_samples, 3, 32, 32, generator=rng).to(device)
    diffusion_model.eval()
    images = diffusion_model.generate(noise, diffusion_steps=diffusion_steps)
    grid = make_grid(images.cpu(), nrow=8, padding=2)
    return to_pil_image(grid)