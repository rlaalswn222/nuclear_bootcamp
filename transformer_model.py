# Transformer 모델


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import numpy as np

# CustomDataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, file_path='./code/processed_fin_data', time_seq=15):
        self.data, self.label = [], []
        file_list = [f for f in os.listdir(file_path) if f.endswith('.csv')]

        for file in file_list:
            df = pd.read_csv(os.path.join(file_path, file))
            df = df.drop(columns=['TICK'], errors='ignore')
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

            features = df.iloc[:, :-1].values
            labels = df.iloc[:, -1].values
            num_rows, num_cols = features.shape

            if num_rows < time_seq:
                continue

            for i in range(num_rows - time_seq + 1):
                self.data.append(features[i:i + time_seq])
                self.label.append(labels[i + time_seq - 1])

        self.data = np.array(self.data)
        self.label = np.array(self.label)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

# PCA 적용 함수
def apply_pca(dataset, n_components=64):
    data = dataset.data.numpy()
    labels = dataset.label.numpy()

    scaler = StandardScaler()
    data = data.reshape(data.shape[0], -1)  # Flatten time_seq * features
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)

    dataset.data = torch.tensor(data_pca, dtype=torch.float32)
    dataset.label = torch.tensor(labels, dtype=torch.long)

    print(f"PCA 적용 완료: 원래 차원 {data.shape[1]} → 축소된 차원 {n_components}")
    return dataset

# Transformer 모델 정의
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward) # 입력 데이터를 dim_feedforward 차원으로 변환
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(dim_feedforward, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, dim_feedforward)
        x = x.unsqueeze(1)  # (batch_size, 1, dim_feedforward) for transformer
        x = x.permute(1, 0, 2)  # (sequence_length=1, batch_size, dim_feedforward)
        x = self.transformer(x)  # (sequence_length=1, batch_size, dim_feedforward)
        x = x.squeeze(0)  # (batch_size, dim_feedforward)
        return self.fc(x)  # (batch_size, num_classes)

# 학습 및 검증 함수
def train_and_validate(train_loader, val_loader, input_dim, num_classes, num_heads, num_layers, dim_feedforward, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel(
        input_dim=input_dim,
        num_classes=num_classes,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    model_save_path = 'transformer_model.pt'

    for epoch in range(epochs):
        # 학습 단계
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = correct_train / total_train

        # 검증 단계
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_accuracy = correct_val / total_val

        # 모델 저장
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch + 1} with validation loss {best_val_loss:.4f}")

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {total_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {total_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Main 함수
def main():
    file_path = './code/processed_fin_data'
    time_seq = 15
    n_components = 64

    dataset = CustomDataset(file_path=file_path, time_seq=time_seq)
    dataset = apply_pca(dataset, n_components=n_components)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    input_dim = n_components  # PCA 차원
    num_classes = 5
    num_heads = 4
    num_layers = 2
    dim_feedforward = 128
    epochs = 20

    train_and_validate(train_loader, val_loader, input_dim, num_classes, num_heads, num_layers, dim_feedforward, epochs)

if __name__ == "__main__":
    main()
