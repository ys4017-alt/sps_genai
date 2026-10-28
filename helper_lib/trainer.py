import torch
from tqdm import tqdm

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