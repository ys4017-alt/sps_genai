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
