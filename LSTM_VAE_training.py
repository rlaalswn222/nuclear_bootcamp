import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os

# CustomDataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, file_path='./test/processed_data', time_seq=15):
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
            num_rows, num_cols = df.shape

            if num_rows < time_seq:
                continue  # 시퀀스 길이보다 적은 행은 스킵

            for i in range(num_rows - time_seq + 1):
                self.data.append(df.iloc[i:i + time_seq, :-1].values)
                self.label.append(df.iloc[i + time_seq - 1, -1])

        # 텐서로 변환
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.long)

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

# VAE 네트워크 정의
class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc_mu = nn.Linear(128, latent_size)
        self.fc_logvar = nn.Linear(128, latent_size)
        self.fc2 = nn.Linear(latent_size, 128)
        self.fc3 = nn.Linear(128, input_size)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# VAE 손실 함수 정의
def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# 학습 루프
if __name__ == "__main__":
    # 하이퍼파라미터 설정
    time_seq = 20  # 시퀀스 길이를 20으로 설정
    batch_size = 64
    epochs = 50
    input_size = 100  # 열(피처)의 개수
    hidden_size = 128
    latent_size = 32
    num_layers = 2
    output_size = 5

    # 데이터셋 및 데이터로더 초기화
    dataset = CustomDataset(file_path='./test/demo', time_seq=time_seq)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 초기화
    lstm_model = LSTM_Network(input_size, hidden_size, num_layers, output_size)
    vae_model = VAE(input_size=input_size, latent_size=latent_size)

    # 옵티마이저 및 손실 함수
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 학습 루프
    for epoch in range(epochs):
        lstm_model.train()
        vae_model.train()
        total_lstm_loss = 0.0
        total_vae_loss = 0.0

        for sample_data, labels in dataloader:
            # VAE 학습
            vae_optimizer.zero_grad()
            recon_x, mu, logvar = vae_model(sample_data)
            vae_loss = vae_loss_function(recon_x, sample_data, mu, logvar)
            vae_loss.backward()
            vae_optimizer.step()
            total_vae_loss += vae_loss.item()

            # LSTM 학습
            lstm_optimizer.zero_grad()
            outputs = lstm_model(sample_data)
            lstm_loss = criterion(outputs, labels)
            lstm_loss.backward()
            lstm_optimizer.step()
            total_lstm_loss += lstm_loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, LSTM Loss: {total_lstm_loss:.4f}, VAE Loss: {total_vae_loss:.4f}")


    # 모델 저장
    torch.save({'lstm_model': lstm_model.state_dict(), 'vae_model': vae_model.state_dict()}, 'LSTM_VAE_model.pt')
    print("모델 저장 완료: LSTM_VAE_model.pt")


    
