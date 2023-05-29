import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../SWE')
from data_driven.aux_function import general_aux

class cnn_lax_flux(nn.Module):
    def __init__(self):
        super(cnn_lax_flux, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(4, 20, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(20, 20, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(20, 40, kernel_size=1, stride=1)
        self.conv4 = nn.Conv1d(40, 10, kernel_size=1, stride=1)
        self.conv5 = nn.Conv1d(10, 3, kernel_size=1, stride=1)

    def forward(self, x):
        # Convolutional layer 1
        x = self.conv1(x)
        x = nn.ReLU()(x)
        # Convolutional layer 2
        x = self.conv2(x)
        x = nn.ReLU()(x)
        # Convolutional layer 3
        x = self.conv3(x)
        x = nn.ReLU()(x)
        # Convolutional layer 4
        x = self.conv4(x)
        x = nn.ReLU()(x)
        # Convolutional layer 5
        x = self.conv5(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        return x
        
# Traning the model
def train(train_loader, val_loader, learning_rates, device, batch_size, maximum_epochs, patience, model_path):
    global_best_val_loss = float('inf')
    best_learning_rate = None
    for learning_rate in learning_rates:
        early_stopping_counter = 0
        local_best_val_loss = float('inf')
        model = cnn_lax_flux().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        for epoch in range(maximum_epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                parameters, labels = batch[0].to(device), batch[1].to(device)
                output = model(parameters)
                loss = criterion(output, labels)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            
            # For each epoch do validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    parameters, labels = batch[0].to(device), batch[1].to(device)
                    output = model(parameters)
                    loss = criterion(output, labels)
                    val_loss += loss.item()
            
            # Calculate the average loss over the epoch
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
            
            # Check if the validation loss is decreasing
            if avg_val_loss < local_best_val_loss:
                local_best_val_loss = avg_val_loss
                early_stopping_counter = 0
                if local_best_val_loss < global_best_val_loss:
                    global_best_val_loss = local_best_val_loss
                    best_learning_rate = learning_rate
                    torch.save(model.state_dict(), model_path)
            else:
                early_stopping_counter += 1
                if early_stopping_counter == patience:
                    print(f"Early stopping at epoch {epoch+1}, with learning rate {learning_rate}")
                    break
    print(f"Best learning rate: {best_learning_rate}, with avg. validation loss of: {global_best_val_loss:.4f}") 
            
# This is to train and store the model, the method load model and compute is for computation.
def main(argv):
    device = torch.device('cpu') # using cpu, the GPU feature is only avalible for certain Nvidia GPU's, can be used for 3-4 times speed up if avalible
    # Hyper parameters
    batch_size = 128
    maximum_epochs = 10000
    learning_rates = [0.1, 0.06, 0.03, 0.01, 0.006, 0.003, 0.001, 0.0006, 0.0003]
    patience = 10
    # Create data loaders
    train_loader, val_loader, test_loader = general_aux.create_data_loaders_from_csv('data_driven/generated_data/lax_friedrich_flux.csv', batch_size, 'cnn_flux')
    train(train_loader, val_loader, learning_rates, device, batch_size, maximum_epochs, patience=patience, model_path='data_driven/models/lax_flux_CNN.pt')
    model = general_aux.load_model(model_path='data_driven/models/lax_flux_CNN.pt', device=device, model_type='cnn_flux')
    general_aux.test(model, test_loader, device)

if __name__ == '__main__':
    main(sys.argv)
    print("Completed training of CNN-Lax-Friedrich-flux")