import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../SWE')
from data_driven.aux_function import general_aux

class ffnn_riemann(nn.Module):
    def __init__(self):
        super(ffnn_riemann, self).__init__()

        self.fc1 = nn.Linear(6, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 10)
        self.fc5 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def train(train_loader, val_loader, learning_rates, device, batch_size, maximum_epochs, patience, model_path):
    global_best_val_loss = float('inf')
    best_learning_rate = None

    for learning_rate in learning_rates:
        early_stopping_counter = 0
        local_best_val_loss = float('inf')

        model = ffnn_riemann().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(maximum_epochs):
            model.train()
            train_loss = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

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

def main(argv):
    try:
        if ('exact' == argv[1]):
            solver = 0 
        elif ('hllc' == argv[1]): 
            solver = 1
        else:
            print("Please specify the Riemann solver to train as the first argument \'exact\' or \'hllc\'")
            sys.exit(1)
    except:
        print("Please specify the Riemann solver to train as the first argument \'exact\' or \'hllc\'")
        sys.exit(1)
    device = torch.device('cpu')  # Change to 'cuda' if available
    # Hyperparameters
    batch_size = 128
    maximum_epochs = 10000
    learning_rates = [0.1, 0.06, 0.03, 0.01, 0.006, 0.003, 0.001, 0.0006, 0.0003]
    patience = 10
    # Create data loaders, train, and test
    train_loader, val_loader, test_loader = general_aux.create_data_loaders_from_csv('data_driven/generated_data/riemann_' + argv[1] + '.csv', batch_size, 'ffnn_riemann')
    train(train_loader, val_loader, learning_rates, device, batch_size, maximum_epochs, patience=patience, model_path=('data_driven/models/riemann_' + argv[1] + '_FFNN.pt'))
    model = general_aux.load_model(model_path=('data_driven/models/riemann_' + argv[1] + '_FFNN.pt'), device=device, model_type='ffnn_riemann')
    general_aux.test(model, test_loader, device)

if __name__ == '__main__':
    main(sys.argv)
    print("Completed training of FFNN for " + sys.argv[1] + " Riemann solver")