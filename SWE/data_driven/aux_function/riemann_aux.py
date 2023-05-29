import numpy as np
import torch
from torch.utils.data import Dataset

# Custom dataset class
class cnn_riemann_dataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data.values, dtype=torch.float32)
         
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = (self.data[index, :6]).view(2, 3).t()
        y = self.data[index, 6:]  
        return x, y

class ffnn_riemann_dataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data.values, dtype=torch.float32)
         
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = (self.data[index, :6])
        y = self.data[index, 6:]  
        return x, y

# Use the trained model to compute the solution to the riemann problem 
def compute(model, W_left, W_right, cnn):
    model.eval()
    inputs = np.concatenate((W_left, W_right), axis=1)
    if cnn == 1:
        inputs = inputs.reshape(-1, 2, 3).swapaxes(1,2)
    inputs = torch.tensor(inputs, dtype=torch.float32)  # Assuming you're using PyTorch
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.numpy() 