import yfinance as yf
import seaborn as sns
import numpy as np
import random
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

class DataProcessing():
    def __init__(self, seq_length, batch_size):
        self.seq_length     = seq_length
        self.batch_size     = batch_size

        # List of all closing prices
        self.closing_prices = self.get_price_data()     

        # Lists of all input sequences and labels
        self.X = []             
        self.y = []             

        # Train, Validation, and Test inputs sequences and labels (torch.tensors)
        self.X_train = None     
        self.y_train = None     
        self.X_val   = None    
        self.y_val   = None     
        self.X_test  = None     
        self.y_test  = None     

        # Normalized Train, Validation, and Test inputs and labels
        self.X_train_norm   = None
        self.y_train_norm   = None
        self.X_val_norm     = None
        self.y_val_norm     = None
        self.X_test_norm    = None
        self.y_test_norm    = None

        # Data loaders
        self.train_loader   = None
        self.val_loader     = None
        self.test_loader    = None

        # Input and Output scalers
        self.in_scaler  = MinMaxScaler()
        self.out_scaler = MinMaxScaler()

    def get_process_data(self):
        self.get_price_data()
        self.separate_input_labels()
        self.separate_train_test()
        return self.create_data_loaders()

    def get_price_data(self):
        # Get data
        df = yf.download("BTC-USD", start="2017-1-1", end="2024-8-1", interval="1d")

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
        shuffled_indices = np.arange(len(self.X))
        random.shuffle(shuffled_indices)

        nr_val  = int(len(self.X) * 0.20)
        nr_test = int(len(self.X) * 0.10)

        val_indices   = shuffled_indices[:nr_val]
        test_indices  = shuffled_indices[nr_val:nr_val+nr_test]
        train_indices = shuffled_indices[nr_test:]

        X_train = torch.tensor(self.X[train_indices], dtype=torch.float32)
        y_train = torch.tensor(self.y[train_indices], dtype=torch.float32)

        X_val   = torch.tensor(self.X[val_indices], dtype=torch.float32)
        y_val   = torch.tensor(self.y[val_indices], dtype=torch.float32)

        X_test  = torch.tensor(self.X[test_indices], dtype=torch.float32)
        y_test  = torch.tensor(self.y[test_indices], dtype=torch.float32)

        # Reshape X,y into (nr_of_samples, seq_length, nr_of_features) -> necessary for LSTM
        self.X_train = torch.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        self.y_train = torch.reshape(y_train, (y_train.shape[0], 1, 1))

        self.X_val = torch.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
        self.y_val = torch.reshape(y_val, (y_val.shape[0], 1, 1))

        self.X_test = torch.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        self.y_test = torch.reshape(y_test, (y_test.shape[0], 1, 1))
    
    def normalize(self, X, y, fit=0):
        # Reshape training data to 2D for normalization
        X_reshaped = X.reshape(-1, X.shape[-1])  # shape: (nr_train_sequences * sequence_length, nr_of_features)
        y_reshaped = y.reshape(-1, y.shape[-1])  # shape: (nr_train_sequences * sequence_length, nr_of_features)

        if fit == 1:
            # Fit scaler to training data and transform training data
            X_normalized = self.in_scaler.fit_transform(X_reshaped)
            y_normalized = self.out_scaler.fit_transform(y_reshaped)
            
            if np.min(X_normalized) < 0: print(f'Value smaller than 0: {np.min(X_normalized)}')
            if np.max(X_normalized) > 1: print(f'Value greater than 1: {np.min(X_normalized)}')

        else:
            # Transform test data
            X_normalized = self.in_scaler.transform(X_reshaped)
            y_normalized = self.out_scaler.transform(y_reshaped)

        # Reshape back to 3D
        X_normalized = X_normalized.reshape(X.shape)
        y_normalized = y_normalized.reshape(y.shape)

        # Convert to PyTorch tensors
        X_norm = torch.tensor(X_normalized, dtype=torch.float32)
        y_norm = torch.tensor(y_normalized, dtype=torch.float32)

        return X_norm, y_norm
    
    def create_data_loaders(self):
        self.X_train_norm, self.y_train_norm  = self.normalize(self.X_train, self.y_train, fit=1)
        self.X_val_norm, self.y_val_norm      = self.normalize(self.X_val, self.y_val, fit=0)
        self.X_test_norm, self.y_test_norm    = self.normalize(self.X_test, self.y_test, fit=0)

        # Combine inputs and labels
        train_dataset  = TensorDataset(self.X_train_norm, self.y_train_norm)
        val_dataset    = TensorDataset(self.X_val_norm, self.y_val_norm)
        test_dataset   = TensorDataset(self.X_test_norm, self.y_test_norm)

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=1, pin_memory=True)
        self.val_loader   = DataLoader(val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=1, pin_memory=True)
        self.test_loader  = DataLoader(test_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=1, pin_memory=True)

        return self.train_loader, self.val_loader, self.test_loader