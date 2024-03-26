import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pdb

# Assuming the DataFrame is loaded into `df` and has a column named 'Open'
# For demonstration, replace the file path with your actual file path
df = pd.read_csv('data/NVDA.csv')
data = df['Open'].values.reshape(-1, 1)  # Reshape for the scaler

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data)

# Convert to PyTorch tensors
data_normalized = torch.FloatTensor(data_normalized).view(-1)

# Create sequences
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw-5):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+5]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

# Define the sequence length
seq_length = 50
sequences = create_inout_sequences(data_normalized, seq_length)

# Create DataLoader
train_loader = DataLoader(sequences, batch_size=16, shuffle=True)

# Define the model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=5):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-5:]


def test_quant_alg_forward():
    # Instantiate the model, define loss function and optimizer
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    pdb.set_trace()
    for i in range(epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            
            pdb.set_trace()
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i%2 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
