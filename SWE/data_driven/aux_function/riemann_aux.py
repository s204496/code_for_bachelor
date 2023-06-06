import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import sys
sys.path.append('../SWE')
from aux_functions import f

# This is the custom loss function for approach1
class custom_loss(nn.Module):
    def __init__(self):
        super(custom_loss, self).__init__()
    
    def forward(self, h_hat, h_l, u_l, h_r, u_r, h_s):
        return torch.mean(((f.f_k_tensor(torch.flatten(h_hat), torch.flatten(h_s), h_l) + f.f_k_tensor(torch.flatten(h_hat), torch.flatten(h_s), h_r) + u_r - u_l)**2) + ((h_hat - h_s)**2))

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predicted, label):
        loss = torch.mean(((predicted - label) ** 2)* 200)
        return loss

#def print_percentage_distribution_of_loss(h_hat, h_l, u_l, h_r, u_r, h_s):
#    first_loss = torch.mean(((f.f_k_tensor(torch.flatten(h_hat), torch.flatten(h_s), h_l) + f.f_k_tensor(torch.flatten(h_hat), torch.flatten(h_s), h_r) + u_r - u_l)**2) + ((h_hat - h_s)**2)).item()
#    second_loss = torch.mean(torch.abs(h_s - h_hat)).item()
#    print("Percentage of loss, first function: " + str(first_loss / (first_loss + second_loss)))
#    print("Percentage of loss, second function: " + str(second_loss / (first_loss + second_loss)))

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
        x = (self.data[index, :4])
        y = self.data[index, 4:]  
        return x, y

# Use the trained model to compute the solution to the riemann problem
def compute(model, input_left, input_right, cnn):
    model.eval()
    inputs = np.concatenate((input_left, input_right), axis=1)
    if cnn == 1:
        inputs = inputs.reshape(-1, 2, 3).swapaxes(1,2)
    inputs = torch.tensor(inputs, dtype=torch.float32)  # Assuming you're using PyTorch
    with torch.no_grad():
        outputs = model(inputs)
    return np.ravel(outputs.numpy())