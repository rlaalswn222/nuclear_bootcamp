
import torch.nn as nn
import torch
from torch.nn.functional import softmax, leaky_relu
import torch.optim as optim

class LSTM_Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        LSTM 기반 시계열 모델 초기화
        Args:
            input_size (int): 입력 특징 수
            hidden_size (int): LSTM의 은닉 상태 크기
            num_layers (int): LSTM 계층 수
            output_size (int): 최종 출력 클래스 수
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # LSTM의 은닉 상태를 입력으로 받아 최종 출력 생성

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM 출력 (batch_size, seq_length, hidden_size)
        final_output = self.fc(lstm_out[:, -1, :])  # 마지막 타임스텝의 출력만 사용 (batch_size, hidden_size)
        return final_output
    
    def net_training(self, training_data, training_target):
        """
        LSTM 모델 학습
        Args:
            training_data (torch.Tensor): 입력 데이터 (배치 크기, 시간 길이, 특징 수)
            training_target (torch.Tensor): 실제 레이블 (배치 크기)
        """
        self.opt = optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer
        self.cri = nn.CrossEntropyLoss()  # 분류를 위한 손실 함수

        predict = self.forward(training_data)  # Forward pass
        loss = self.cri(predict, training_target)  # Calculate loss

        self.opt.zero_grad()  # Zero gradients
        loss.backward()       # Backpropagation
        self.opt.step()       # Update weights

        # print(f"Training step completed. Loss: {loss.item():.4f}")

        return loss.item()
    
    def calculate_accuracy(self, inputs, targets):
        outputs = self.forward(inputs)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        accurary = (correct / targets.size(0)) * 100
        return accurary


if __name__ == "__main__":
     # 모델 초기화
    input_size = 3     # 입력 특징 수
    hidden_size = 20   # LSTM 은닉 상태 크기
    num_layers = 2     # LSTM 계층 수
    output_size = 5    # 출력 클래스 수

    model = LSTM_Network(input_size, hidden_size, num_layers, output_size)

    for epoch in range(50):
        inputs = torch.tensor([
            [[0.5, -0.2, 0.3], [0.1, 0.8, 0.1], [0.2, 0.2, 0.4]], 
            [[0.2, 0.1, 0.7], [-0.2, 0.8, 0.6], [0.5, 0.3, 0.1]]
        ], dtype=torch.float32) # 3D (2, 3, 3)

        # targets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long) # Batch 3
        targets = torch.tensor([0, 1], dtype=torch.long)  # 크기: (2,)

        # 학습
        training_loss = model.net_training(inputs, targets)
        accuracy = model.calculate_accuracy(inputs, targets)
        print(f"Epoch {epoch + 1} | Loss: {training_loss:.4f} | Accuracy: {accuracy:.2f} %")
