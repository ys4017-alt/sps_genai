import random

import torch
from tqdm import tqdm

from helper_lib.generator import langevin_sample

def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)
    # TODO: run several iterations of the training loop (based on epochs parameter) and return the model.
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct, running_total = 0,0
        model.train()
        train_loader_with_progress = tqdm(
            iterable=data_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}"
        )
        for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # log data for tracking
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()
            
            if batch_number % 100 == 99:
                        train_loader_with_progress.set_postfix(
                            {
                                "avg accuracy": f"{running_correct/running_total:.3f}",
                                "avg loss": f"{running_loss/(batch_number+1):.4f}",
                            }
                        )
        epoch_loss = running_loss / len(data_loader)
        epoch_accuracy = 100* running_correct / running_total
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%")
    print("finished traning")
    return model

def train_vae_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    # TODO: run several iterations of the training loop (based on epochs parameter) and return the model
    return model

def train_gan(generator, discriminator, data_loader, criterion, g_optimizer, d_optimizer, device='cpu', epochs=10):
    generator.to(device)
    discriminator.to(device)
    for epoch in range(epochs):
        running_d_loss, running_g_loss = 0.0, 0.0
        generator.train()
        discriminator.train()
        train_loader_with_progress = tqdm(
            iterable=data_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}"
        )
        for batch_number, (real_images, _) in enumerate(train_loader_with_progress):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            z= torch.randn(batch_size, generator.z_dim, device=device)
            fake_images = generator(z)
            
            #discriminator
            d_optimizer.zero_grad()
            
            d_loss = criterion(discriminator(real_images), real_labels) + \
                criterion(discriminator(fake_images.detach()), fake_labels)
            d_loss.backward()
            d_optimizer.step()
            
            #generator
            g_optimizer.zero_grad()
            g_loss = criterion(discriminator(fake_images), real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()
            
            if batch_number % 100 == 99:
                train_loader_with_progress.set_postfix(
                    {
                        "D loss": f"{running_d_loss/(batch_number+1):.4f}",
                        "G loss": f"{running_g_loss/(batch_number+1):.4f}",
                    }
                )
        print(f"Epoch {epoch+1}: D loss={running_d_loss/len(data_loader):.4f}, G loss={running_g_loss/len(data_loader):.4f}")
    print("finished training")
    return generator, discriminator

#A4
def train_ebm(model, data_loader, optimizer, device='cpu', epochs=5, alpha=0.1, steps=60, step_size=10.0, noise=0.005):
    model.to(device)
    # replay buffer: Langevin chains continue from earlier samples instead of restarting from noise
    buffer = [torch.rand(1, 3, 32, 32, device=device) * 2 - 1 for _ in range(128)]
    for epoch in range(epochs):
        running_loss, running_real, running_fake = 0.0, 0.0, 0.0
        train_loader_with_progress = tqdm(
            iterable=data_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}"
        )
        for batch_number, (real_images, _) in enumerate(train_loader_with_progress):
            real_images = real_images.to(device)
            real_images = (real_images + torch.randn_like(real_images) * noise).clamp(-1.0, 1.0)
            batch_size = real_images.size(0)

            # ~5% of chains restart from uniform noise, the rest continue from the buffer
            n_new = sum(random.random() < 0.05 for _ in range(batch_size))
            start = torch.cat(
                [torch.rand(n_new, 3, 32, 32, device=device) * 2 - 1]
                + random.choices(buffer, k=batch_size - n_new)
            )
            fake_images = langevin_sample(model, start, steps=steps, step_size=step_size, noise_std=noise)
            buffer = (list(fake_images.split(1)) + buffer)[:8192]

            model.train()
            real_out, fake_out = model(torch.cat([real_images, fake_images])).split(batch_size)
            cdiv_loss = real_out.mean() - fake_out.mean()
            reg_loss = alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
            loss = cdiv_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            running_loss += loss.item()
            running_real += real_out.mean().item()
            running_fake += fake_out.mean().item()
            if batch_number % 50 == 49:
                train_loader_with_progress.set_postfix(
                    {
                        "loss": f"{running_loss/(batch_number+1):.4f}",
                        "E(real)": f"{running_real/(batch_number+1):.4f}",
                        "E(fake)": f"{running_fake/(batch_number+1):.4f}",
                    }
                )
        print(f"Epoch {epoch+1}: loss={running_loss/len(data_loader):.4f}, " f"E(real)={running_real/len(data_loader):.4f}, E(fake)={running_fake/len(data_loader):.4f}")
    print("finished training")
    return model

#A4
def train_diffusion(model, data_loader, optimizer, loss_fn, device='cpu', epochs=10):
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        train_loader_with_progress = tqdm(
            iterable=data_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}"
        )
        for batch_number, (images, _) in enumerate(train_loader_with_progress):
            running_loss += model.train_step(images.to(device), optimizer, loss_fn)
            if batch_number % 100 == 99:
                train_loader_with_progress.set_postfix({"avg loss": f"{running_loss/(batch_number+1):.4f}"})
        print(f"Epoch {epoch+1}: loss={running_loss/len(data_loader):.4f}")
    print("finished training")
    return model