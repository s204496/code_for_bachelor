import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import sys
import time
sys.path.append('../SWE')
from data_driven.aux_function import general_aux, riemann_aux
from aux_functions import f, plotter

class ffnn_riemann_shallow(nn.Module):
    def __init__(self):
        super(ffnn_riemann_shallow, self).__init__()

        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 10)
        self.fc5 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class ffnn_riemann_deep(nn.Module):
    def __init__(self):
        super(ffnn_riemann_deep, self).__init__()

        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 20)
        self.fc6 = nn.Linear(20, 10)
        self.fc7 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = self.fc7(x)
        return x

def train(train_loader, val_loader, test_loader, learning_rates, device, batch_size, maximum_epochs, patience, model_path, approach, shallow):
    global_best_val_loss = float('inf')
    best_learning_rate = None
    best_lr_l_evalution = [[], []]
    test_loss_best_model = []

    for learning_rate in learning_rates:
        test_error_pr_epoch = [[],[]]
        early_stopping_counter = 0
        local_best_val_loss = float('inf')
        best_test_error = float('inf')

        if (shallow):
            model = ffnn_riemann_shallow().to(device)
        else:
            model = ffnn_riemann_deep().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if (approach == 0):
            criterion = nn.MSELoss()
        elif (approach == 1):
            criterion = riemann_aux.custom_loss()

        for epoch in range(maximum_epochs):
            model.train()
            train_loss = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if approach == 0:
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, inputs[:,0], inputs[:,1], inputs[:,2], inputs[:,3], labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            normal_test_loss = general_aux.test(model, test_loader, device, 0)
            custom_test_loss = general_aux.test(model, test_loader, device, 1)
            test_error_pr_epoch[0].append(normal_test_loss)
            test_error_pr_epoch[1].append(custom_test_loss)

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    if approach == 0:
                        loss = criterion(outputs, labels)
                    else:
                        loss = criterion(outputs, inputs[:,0], inputs[:,1], inputs[:,2], inputs[:,3], labels)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
                
            if avg_val_loss < local_best_val_loss:
                local_best_val_loss = avg_val_loss
                early_stopping_counter = 0

                if local_best_val_loss < global_best_val_loss:
                    global_best_val_loss = local_best_val_loss
                    best_learning_rate = learning_rate
                    best_lr_l_evalution = test_error_pr_epoch
                    best_test_error = normal_test_loss
                    test_loss_best_model = [normal_test_loss, custom_test_loss]
                    torch.save(model.state_dict(), model_path)
            else:
                early_stopping_counter += 1
                if early_stopping_counter == patience:
                    print(f"Early stopping at epoch {epoch+1}, with learning rate {learning_rate}")
                    break
            if epoch % 200 == 5:
                print(f"Epoch: {epoch+1}, Training loss: {avg_train_loss:.5f}, Validation loss: {avg_val_loss:.5f}")
        print(f"Learning rate: {learning_rate}, with best avg. validation loss of: {local_best_val_loss:.5f}")
    print(f"Best learning rate: {best_learning_rate}, with avg. validation loss of: {global_best_val_loss:.5f}")
    print(f'Test error for best validation model: {best_test_error:.5f}')
    return best_lr_l_evalution

def main(argv):
    solver = 0
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
            samples_str = '1d6M'
        else:
            print('Please specify the number of samples to generate as first argument\n0: for 25.000 samples\n1: for 200.000 samples\n2: for 1.600.000 samples')
            sys.exit(1)
    except:
        print('Please specify the number of samples to generate as first argument\n0: for 25.000 samples\n1: for 200.000 samples\n2: for 1.600.000 samples')
        sys.exit(1)
    try:
        if (argv[2] == 'exact'):
            pass
        elif (argv[2] == 'hllc'):
            solver = 1
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
    learning_rates = [0.01, 0.006, 0.003, 0.001, 0.0006]
    patience = 150
    # Create data loaders, train, and test
    train_loader, val_loader, test_loader = general_aux.create_data_loaders_from_csv('data_driven/generated_data/riemann_' + argv[2] + '_' + samples_str + '.csv', batch_size, 'ffnn_riemann')
    # Train 3 shallow and  
    models_testerror = []
    models_testerror.append(train(train_loader, val_loader, test_loader, learning_rates, device, batch_size, maximum_epochs, patience=patience, model_path=('data_driven/models/riemann_' + argv[2] + '_' + samples_str + '_FFNN_shallow.pt'), approach=0, shallow=True))
    #models_testerror.append(train(train_loader, val_loader, test_loader, learning_rates, device, batch_size, maximum_epochs, patience=patience, model_path=('data_driven/models/riemann_FFNN_deep.pt'), approach=0, shallow=False))
    #models_testerror.append(train(train_loader, val_loader, test_loader, learning_rates, device, batch_size, maximum_epochs, patience=patience, model_path=('data_driven/models/riemann_FFNN_shallow_custom_loss.pt'), approach=1, shallow=True))
    ## Plot the results
    #plotter.plot_experiment_1(models_testerror)

if __name__ == '__main__':
    main(sys.argv)
    print("Completed training of FFNN for Riemann solver")
