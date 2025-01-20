import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import numpy as np

# CustomDataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, file_path='./code/processed_fin_data', time_seq=5):
        """
        시계열 데이터를 생성하는 데이터셋 클래스
        Args:
            file_path (str): CSV 파일 경로가 포함된 폴더
            time_seq (int): 시퀀스 길이
        """
        self.data, self.label = [], []
        file_list = [f for f in os.listdir(file_path) if f.endswith('.csv')]

        for file in file_list:
            df = pd.read_csv(os.path.join(file_path, file))
            
            # TICK 열 제거 및 NaN/Inf 값 처리
            df = df.drop(columns=['TICK'], errors='ignore')
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            # 데이터와 라벨 분리
            features = df.iloc[:, :-1].values  # 2열부터 마지막 열 전까지
            labels = df.iloc[:, -1].values  # 마지막 열

            num_rows, num_cols = features.shape
            if num_rows < time_seq:
                continue

            # 시계열 데이터 생성
            for i in range(num_rows - time_seq + 1):
                self.data.append(features[i:i + time_seq])
                self.label.append(labels[i + time_seq - 1])
        
        # 리스트를 numpy 배열로 변환 후 텐서 변환
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.label = torch.tensor(np.array(self.label), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

# LSTM 네트워크 정의
class LSTM_Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Network, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# 학습 루프
if __name__ == "__main__":
    # 하이퍼파라미터 설정
    time_seq = 20  # 시퀀스 길이
    batch_size = 64
    epochs = 50
    input_size = 117  # 열(피처)의 개수 (TICK 제외)
    hidden_size = 128
    num_layers = 2
    output_size = 5  # 클래스 개수

    # 데이터셋 및 데이터로더 초기화
    dataset = CustomDataset(file_path='./code/processed_fin_data', time_seq=time_seq)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 초기화
    lstm_model = LSTM_Network(input_size, hidden_size, num_layers, output_size)

    # 옵티마이저 및 손실 함수
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 학습 루프
    for epoch in range(epochs):
        lstm_model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        mse_loss = 0.0
        mae_loss = 0.0

        for sample_data, labels in dataloader:
            optimizer.zero_grad()
            outputs = lstm_model(sample_data)

            # 손실 계산
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계 계산
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            mse_loss += nn.MSELoss()(outputs, torch.nn.functional.one_hot(labels, num_classes=output_size).float()).item()
            mae_loss += torch.abs(outputs - torch.nn.functional.one_hot(labels, num_classes=output_size).float()).mean().item()

        accuracy = correct_predictions / total_samples
        mse_loss /= len(dataloader)
        mae_loss /= len(dataloader)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, MSE: {mse_loss:.4f}, MAE: {mae_loss:.4f}")

    # 모델 저장
    torch.save(lstm_model.state_dict(), 'LSTM_model.pt')
    print("모델 저장 완료: LSTM_model.pt")
