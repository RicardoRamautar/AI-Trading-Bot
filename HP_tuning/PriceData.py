import yfinance as yf
import seaborn as sns
import numpy as np
import random
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

class DataProcessing():
    def __init__(self, seq_length, batch_size):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.closing_prices = self.get_price_data()
        self.X = []
        self.y = []
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.X_norm = None
        self.y_norm = None
        self.train_loader = None
        self.val_loader = None

    def get_process_data(self):
        self.get_price_data()
        self.separate_input_labels()
        return self.separate_train_test()

    def get_price_data(self):
        # Get data
        df = yf.download("BTC-USD", start="2017-1-1", end="2024-1-1", interval="1d")

        # Get list of closing prices
        closing_prices = list(df.iloc[:, 3])
        return closing_prices
    
    def show_spread(self):
        sns.set_style('darkgrid')
        sns.kdeplot(np.array(self.closing_prices))

    def separate_input_labels(self):
        X, y = [], []
        for i in range(len(self.closing_prices)-self.seq_length):
            X.append(self.closing_prices[i:i+self.seq_length])
            y.append(self.closing_prices[i+self.seq_length])

        self.X = np.array(X)
        self.y = np.array(y)
    
    def separate_train_test(self):
        test_indices  = random.sample(range(len(self.X)), 300)
        X_test = torch.tensor(self.X[test_indices], dtype=torch.float32)
        y_test = torch.tensor(self.y[test_indices], dtype=torch.float32)

        train_indices = [i for i in range(len(self.X)) if i not in test_indices]
        X_train = torch.tensor(self.X[train_indices], dtype=torch.float32)
        y_train = torch.tensor(self.y[train_indices], dtype=torch.float32)

        # Reshape X,y into (nr_of_samples, seq_length, nr_of_features) -> necessary for LSTM
        self.X_test = torch.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        self.y_test = torch.reshape(y_test, (y_test.shape[0], 1, 1))

        self.X_train = torch.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        self.y_train = torch.reshape(y_train, (y_train.shape[0], 1, 1))

        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def normalize(self, X, y, input_scaler=MinMaxScaler(), 
                  output_scaler=MinMaxScaler(), fit=0):
        # Reshape training data to 2D for normalization
        X_reshaped = X.reshape(-1, X.shape[-1])  # shape: (nr_train_sequences * sequence_length, nr_of_features)
        y_reshaped = y.reshape(-1, y.shape[-1])  # shape: (nr_train_sequences * sequence_length, nr_of_features)

        if fit == 1:
            # Fit scaler to training data and transform training data
            X_normalized = input_scaler.fit_transform(X_reshaped)
            y_normalized = output_scaler.fit_transform(y_reshaped)
            
            if np.min(X_normalized) < 0: print(f'Value smaller than 0: {np.min(X_normalized)}')
            if np.max(X_normalized) > 1: print(f'Value greater than 1: {np.min(X_normalized)}')

        else:
            # Transform test data
            X_normalized = input_scaler.transform(X_reshaped)
            y_normalized = output_scaler.transform(y_reshaped)

        # Reshape back to 3D
        X_normalized = X_normalized.reshape(X.shape)
        y_normalized = y_normalized.reshape(y.shape)

        # Convert to PyTorch tensors
        self.X_norm = torch.tensor(X_normalized, dtype=torch.float32)
        self.y_norm = torch.tensor(y_normalized, dtype=torch.float32)

        return self.X_norm, self.y_norm, input_scaler, output_scaler
    
    def create_fold_sets(self, train_indices, val_indices):
        # Normalize Training and Validation sets
        X_train_fold, y_train_fold  = self.X_train[train_indices], self.y_train[train_indices]
        X_val_fold, y_val_fold      = self.X_train[val_indices], self.y_train[val_indices]

        X_train_fold, y_train_fold, in_scaler, out_scaler = self.normalize(X_train_fold, y_train_fold, fit=1)
        X_val_fold, y_val_fold, _, _ = self.normalize(X_val_fold, y_val_fold, in_scaler, out_scaler, fit=0)

        # Combine inputs and labels
        train_dataset_fold  = TensorDataset(X_train_fold, y_train_fold)
        val_dataset_fold    = TensorDataset(X_val_fold, y_val_fold)

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset_fold, batch_size=self.batch_size,
                                shuffle=True, num_workers=1, pin_memory=True)
        self.val_loader   = DataLoader(val_dataset_fold, batch_size=self.batch_size,
                                shuffle=False, num_workers=1, pin_memory=True)

        return self.train_loader, self.val_loader