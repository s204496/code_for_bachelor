import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class godunov_flux_dataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data.values, dtype=torch.float32)
         
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = (self.data[index, :4])
        y = self.data[index, 4:]  
        return x, y

# Use the trained model to compute the flux 
def compute(model, W_left, W_right):
    model.eval()
    inputs = np.concatenate((W_left, W_right), axis=1)
    inputs = torch.tensor(inputs, dtype=torch.float32)  
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.numpy() 

