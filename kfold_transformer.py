# kfold transformer

from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import pandas as pd
import os

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
    data = data.reshape(data.shape[0], -1)
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
        self.embedding = nn.Linear(input_dim, dim_feedforward)
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
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = x.mean(dim=1)
        return self.fc(x)

# K-Fold 학습 및 검증
def train_and_validate_kfold(dataset, input_dim, time_seq, num_classes, num_heads, num_layers, dim_feedforward, epochs=20, k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{k}")

        # 데이터 분리
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

        # 모델 초기화
        model = TransformerModel(
            input_dim=input_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 학습
        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            total_train_loss, correct_train, total_train = 0.0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # 입력 데이터 차원 추가 (batch_size, 1, input_dim)
                inputs = inputs.unsqueeze(1)

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

            # 검증
            model.eval()
            total_val_loss, correct_val, total_val = 0.0, 0, 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # 입력 데이터 차원 추가 (batch_size, 1, input_dim)
                    inputs = inputs.unsqueeze(1)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct_val += (preds == labels).sum().item()
                    total_val += labels.size(0)

            val_accuracy = correct_val / total_val

            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Train Loss: {total_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Val Loss: {total_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                torch.save(model.state_dict(), f"transformer_fold{fold + 1}.pt")
                print(f"Model saved for fold {fold + 1} at epoch {epoch + 1}")

        results.append((fold + 1, best_val_loss, val_accuracy))

    print("K-Fold Results:")
    for fold, loss, acc in results:
        print(f"Fold {fold}: Best Val Loss = {loss:.4f}, Val Accuracy = {acc:.4f}")

# Main 함수
def main():
    file_path = './code/processed_fin_data'
    time_seq = 15
    n_components = 64

    dataset = CustomDataset(file_path=file_path, time_seq=time_seq)
    dataset = apply_pca(dataset, n_components=n_components)

    input_dim = n_components
    num_classes = 5
    num_heads = 4
    num_layers = 2
    dim_feedforward = 128
    epochs = 20
    k = 5

    train_and_validate_kfold(dataset, input_dim, time_seq, num_classes, num_heads, num_layers, dim_feedforward, epochs, k)

if __name__ == "__main__":
    main()
