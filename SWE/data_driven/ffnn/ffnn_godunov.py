import pandas as pd
import torch
import torch.nn as nn
import sys
sys.path.append('../SWE')
from data_driven.aux_function import general_aux
from aux_functions import plotter

class ffnn_godunov(nn.Module):
    def __init__(self):
        super(ffnn_godunov, self).__init__()

        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 10)
        self.fc5 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = torch.sin(self.fc2(x))
        x = torch.sin(self.fc3(x))
        x = torch.sin(self.fc4(x))
        x = self.fc5(x)
        return x

def train(train_loader, val_loader, test_loader, learning_rates, device, maximum_epochs, patience, model_path):
    global_best_val_loss = float('inf')
    best_test_error = float('inf')
    best_learning_rate = None
    error_pr_epoch_best_lr = [[],[]] # store train and test error pr epoch for best learning rate

    for learning_rate in learning_rates:
        early_stopping_counter = 0
        error_pr_epoch = [[],[]] # store train and test error pr epoch
        local_best_val_loss = float('inf')

        model = ffnn_godunov().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(maximum_epochs):
            model.train()
            train_loss = 0

            for inputs, labels in train_loader:
                inputs_model, labels_model = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs_model)
                loss = criterion(outputs, labels_model)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            test_error = general_aux.test(model, test_loader, device, 0)
            error_pr_epoch[0].append(test_error)

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            error_pr_epoch[1].append(avg_train_loss)

                
            if avg_val_loss < local_best_val_loss:
                local_best_val_loss = avg_val_loss
                early_stopping_counter = 0

                if local_best_val_loss < global_best_val_loss:
                    global_best_val_loss = local_best_val_loss
                    best_learning_rate = learning_rate
                    best_test_error = test_error 
                    error_pr_epoch_best_lr = error_pr_epoch 
                    torch.save(model.state_dict(), model_path)
            else:
                early_stopping_counter += 1
                if early_stopping_counter == patience:
                    print(f"Early stopping at epoch {epoch+1}, with learning rate {learning_rate}")
                    break
            if epoch % 200 == 5:
                print(f"Epoch: {epoch+1}, Training loss: {avg_train_loss:.4f}, Validation loss: {avg_val_loss:.4f}")
        print(f"Learning rate: {learning_rate}, with best avg. validation loss of: {local_best_val_loss:.4f}")
    print(f"Best learning rate: {best_learning_rate}, with avg. validation loss of: {global_best_val_loss:.4f}")
    print(f'Test error for best validation model: {best_test_error:.4f}')
    return error_pr_epoch_best_lr, best_learning_rate

def main(argv):
    try:
        sample_choice = int(argv[1])
        if (sample_choice == 0):
            samples = 25000
            samples_str = '25k'
        elif (sample_choice == 1):
            samples = 200000
            samples_str = '200k'
        elif (sample_choice == 2):
            samples = 1600000
            samples_str = '1.6M'
        else:
            print('Please specify the number of samples to generate as first argument\n0: for 25.000 samples\n1: for 200.000 samples\n2: for 1.600.000 samples')
            sys.exit(1)
    except:
        print('Please specify the number of samples to generate as first argument\n0: for 25.000 samples\n1: for 200.000 samples\n2: for 1.600.000 samples')
        sys.exit(1)
    try:
        if (argv[2] == 'exact' or argv[2] == 'hllc'):
            pass
        else:
            print('Please specify the \'exact\' or \'hllc\' as second argument')
            sys.exit(1)
    except:
        print('Please specify the \'exact\' or \'hllc\' as second argument')
        sys.exit(1)
    device = torch.device('cpu')  # Change to 'cuda' if available
    # Hyperparameters
    batch_size = 128
    maximum_epochs = 20000
    learning_rates = [0.003, 0.001, 0.0006, 0.0003, 0.0001]
    patience = 150
    # Create data loaders, train, and test
    train_loader, val_loader, test_loader = general_aux.create_data_loaders_from_csv(('data_driven/generated_data/godunov_flux_' + argv[2] + '_' + samples_str  + '.csv'), batch_size, 'godunov_flux')
    models_train_test_error, best_learning_rate = train(train_loader, val_loader, test_loader, learning_rates, device, maximum_epochs, patience=patience, model_path=('data_driven/models/godunov_flux_' + argv[2] + '_' + samples_str  + '.pt'))
    # Plot the results
    plotter.plot_experiment_2(models_train_test_error, best_learning_rate, samples, (argv[2] + '_' + samples_str))

if __name__ == '__main__':
    main(sys.argv)
    print("Completed training of FFNN for Godunov")