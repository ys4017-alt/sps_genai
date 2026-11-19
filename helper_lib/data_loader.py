import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



def get_data_loader(data_dir, batch_size=32, train=True, image_size=32, dataset="CIFAR10", normalize=False):
  # TODO: create the data loader
  #prepare the data
  if dataset == "MNIST":
    transform = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
  else:
    steps = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if normalize:
      steps.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # [0,1] -> [-1,1]
    transform = transforms.Compose(steps)
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

  if train:
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  else:
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  return loader