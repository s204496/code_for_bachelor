import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import sys

from data_driven.ffnn import ffnn_godunov
sys.path.append('../SWE')
from data_driven.cnn import cnn_riemann 
from data_driven.aux_function import lax_godunov_flux_aux, riemann_aux 
from data_driven.ffnn import ffnn_godunov, ffnn_riemann

# Load the trained model
def load_model(model_path, device, model_type):
    match model_type:
        case 'cnn_riemann':
            model = cnn_riemann.cnn_riemann().to(device)
        case 'ffnn_riemann':
            model = ffnn_riemann.ffnn_riemann_shallow().to(device)
        case 'godunov_flux':
            model = ffnn_godunov.ffnn_godunov().to(device)
    model.load_state_dict(torch.load(model_path))
    return model

# test the loaded model
def test(model, test_loader, device, loss_type):
    model.eval()
    test_loss = 0.0
    if loss_type == 0 or loss_type == 3:
        criterion = nn.MSELoss()
    elif loss_type == 1:
        criterion = riemann_aux.custom_loss() 
    elif loss_type == 4:
        criterion = lax_godunov_flux_aux.custom_loss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if loss_type == 0:
                loss = criterion(outputs, labels)
            elif loss_type == 1:
                loss = criterion(outputs, inputs[:,0], inputs[:,1], inputs[:,2], inputs[:,3], labels)
            test_loss += loss.item()
    return test_loss / len(test_loader)

# Create the data loaders
def create_data_loaders_from_csv(path, batch_size, model_type):
    data = pd.read_csv(path)
    train_data, val_test_data = train_test_split(data, test_size=0.4, random_state=0)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=0)
    train_dataset, val_dataset, test_dataset = None, None, None
    match model_type:
        case 'cnn_riemann':
            train_dataset = riemann_aux.cnn_riemann_dataset(train_data)
            val_dataset = riemann_aux.cnn_riemann_dataset(val_data)
            test_dataset = riemann_aux.cnn_riemann_dataset(test_data)
        case 'ffnn_riemann':
            train_dataset = riemann_aux.ffnn_riemann_dataset(train_data)
            val_dataset = riemann_aux.ffnn_riemann_dataset(val_data)
            test_dataset = riemann_aux.ffnn_riemann_dataset(test_data)
        case 'godunov_flux':
            train_dataset = lax_godunov_flux_aux.godunov_flux_dataset(train_data)
            val_dataset = lax_godunov_flux_aux.godunov_flux_dataset(val_data)
            test_dataset = lax_godunov_flux_aux.godunov_flux_dataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader




