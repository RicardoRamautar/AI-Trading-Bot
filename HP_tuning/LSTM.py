import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, nr_fc_features, device):
        super(LSTM, self).__init__()

        self.device         = device
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc1 = nn.Linear(hidden_size, nr_fc_features)
        self.fc2 = nn.Linear(nr_fc_features, output_size)

        self.relu = nn.ReLU()

    def forward(self, X):
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(self.device)

        output, (hn, cn) = self.lstm(X, (h0, c0))

        # Only use the final hidden state of the final layer as input to the FF layers
        hn = hn[-1]

        out = self.relu(hn)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out