from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import pandas as pd
import os

# CustomDataset 클래스 정의 (PCA 적용 전)
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
    """
    데이터셋에 PCA 적용
    Args:
        dataset (CustomDataset): 데이터셋 객체
        n_components (int): PCA 차원 수
    Returns:
        Tensor: PCA가 적용된 데이터셋
    """
    # 데이터와 라벨 분리
    data = dataset.data.numpy()  # Numpy 변환
    labels = dataset.label.numpy()

    # 데이터 표준화 (Standardization)
    scaler = StandardScaler()
    data = data.reshape(data.shape[0], -1)  # (batch, time_seq * features)
    data_scaled = scaler.fit_transform(data)

    # PCA 적용
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)

    # PCA 변환 후 텐서로 변환
    dataset.data = torch.tensor(data_pca, dtype=torch.float32)
    dataset.label = torch.tensor(labels, dtype=torch.long)

    print(f"PCA 적용 완료: 원래 차원 {data.shape[1]} → 축소된 차원 {n_components}")
    return dataset

# LSTM 모델 정의
class LSTM_Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Network, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# 학습 및 검증 함수
def train_and_validate(train_loader, val_loader, input_size, hidden_size, num_layers, output_size, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 초기화
    model = LSTM_Network(input_size, hidden_size, num_layers, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # 학습 단계
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # LSTM 입력 데이터의 차원을 (batch, time_seq, input_size)로 변환
            inputs = inputs.view(inputs.size(0), -1, input_size)

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
                inputs = inputs.view(inputs.size(0), -1, input_size)

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

# Main 함수
def main():
    # 데이터 준비 및 PCA 적용
    file_path = './code/processed_fin_data'
    time_seq = 15
    n_components = 64  # PCA 차원 수

    dataset = CustomDataset(file_path=file_path, time_seq=time_seq)
    dataset = apply_pca(dataset, n_components=n_components)

    # 데이터 분리 및 DataLoader 준비
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 하이퍼파라미터 설정
    input_size = n_components  # PCA 차원
    hidden_size = 128
    num_layers = 2
    output_size = 5  # 클래스 수
    epochs = 20

    # 학습 및 검증
    train_and_validate(train_loader, val_loader, input_size, hidden_size, num_layers, output_size, epochs=epochs)

# 진입점
if __name__ == "__main__":
    main()
