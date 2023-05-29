import numpy as np
import torch
from torch.utils.data import Dataset

# Custom dataset class
class cnn_flux_dataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data.values, dtype=torch.float32)
         
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = (self.data[index, :8]).view(2, 4).t()
        y = self.data[index, 8:]  
        return x, y

class ffnn_flux_dataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data.values, dtype=torch.float32)
         
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = (self.data[index, :8])
        y = self.data[index, 8:]  
        return x, y

# Use the trained model to compute the solution to the riemann problem 
def compute(model, W_left, W_right, delta_x, delta_t, model_type):
    model.eval()
    delta_x_array = np.full((len(W_left), 1), delta_x)
    delta_t_array = np.full((len(W_left), 1), delta_t)
    inputs = np.concatenate((W_left, delta_x_array, W_right, delta_t_array), axis=1)
    if model_type == 1:
        inputs = inputs.reshape(-1, 2, 4).swapaxes(1,2)
    inputs = torch.tensor(inputs, dtype=torch.float32)  # Assuming you're using PyTorch
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.numpy() 

