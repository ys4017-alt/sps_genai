import os

import torch


def save_model(model, path):
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  checkpoint = {
    'model_state_dict': model.state_dict(),
  }
  torch.save(checkpoint, path)
    


def load_model(model, path, device='cpu'):
  checkpoint = torch.load(path, map_location=device)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.to(device)
  return model
    
    
