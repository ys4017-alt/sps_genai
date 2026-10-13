"""Reusable helper library for the neural network projects in this course.

Module 4 Activity — Neural Network Reusable Code.
Encapsulates data loading, model construction, training and evaluation so the
same code is not duplicated across notebooks.

Usage (from the activity guide):

    from helper_lib.data_loader import get_data_loader
    from helper_lib.trainer import train_model
    from helper_lib.evaluator import evaluate_model
    from helper_lib.model import get_model
    from helper_lib.utils import save_model

Module 5 Activity — Adding VAE to Helper Library:

    from helper_lib.data_loader import get_data_loader
    from helper_lib.trainer import train_vae_model
    from helper_lib.model import get_model
    from helper_lib.generator import generate_samples
    import torch
    import torch.optim as optim

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = get_data_loader('data/train', batch_size=64)
    test_loader = get_data_loader('data/test', batch_size=64, train=False)

    vae = get_model("VAE")
    criterion = # TODO: add a loss function
    optimizer = optim.Adam(vae.parameters(), lr=0.001)

    train_vae_model(vae, train_loader, criterion, optimizer, epochs=5)
    generate_samples(vae, device, num_samples=10)
"""
