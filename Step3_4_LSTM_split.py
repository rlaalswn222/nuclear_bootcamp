import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

class LSTM_Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_output = self.fc(lstm_out[:, -1, :])
        return final_output

    def train_model(self, train_data, train_labels, optimizer, criterion):
        self.train()
        optimizer.zero_grad()
        outputs = self.forward(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, val_data, val_labels):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(val_data)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == val_labels).sum().item()
            accuracy = (correct / val_labels.size(0)) * 100
        return accuracy

# process 전처리함수
def load_and_preprocess_data(data_folder, minmax_path):
    # Load Min-Max scaling parameters
    minmax_df = pd.read_csv(minmax_path)

    # Combine all CSV files into a single DataFrame
    all_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.csv')]
    data_list = [pd.read_csv(file) for file in all_files]
    data = pd.concat(data_list, ignore_index=True)

    # Assume the last column is the label
    data['label'] = (data.index % 4)  # Example: cyclic labels [0, 1, 2, 3]

    # Match the Min-Max scaling parameters to the features
    minmax_df = minmax_df[data.columns[:-1]]  # Ensure columns match
    para_min = np.array(minmax_df.iloc[0])
    para_max = np.array(minmax_df.iloc[1])

    # Features and labels
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Min-Max Scaling
    X_train = (X_train - para_min) / (para_max - para_min)
    X_val = (X_val - para_min) / (para_max - para_min)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, X_train.shape[1])
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32).view(-1, 1, X_val.shape[1])
    y_val = torch.tensor(y_val, dtype=torch.long)

    return X_train, X_val, y_train, y_val, X_train.shape[2]

# Main execution
if __name__ == "__main__":
    data_folder = './code/processed_data'
    minmax_path = './code/minmax_scaled_data.csv'

    # Load and preprocess data
    X_train, X_val, y_train, y_val, input_size = load_and_preprocess_data(data_folder, minmax_path)

    # Model parameters
    hidden_size = 20
    num_layers = 2
    output_size = 4

    # Initialize model, optimizer, and loss function
    model = LSTM_Network(input_size, hidden_size, num_layers, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = 50
    for epoch in range(epochs):
        train_loss = model.train_model(X_train, y_train, optimizer, criterion)
        val_accuracy = model.evaluate(X_val, y_val)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
