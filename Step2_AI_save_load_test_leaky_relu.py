import torch.nn as nn
import torch
from torch.nn.functional import softmax, leaky_relu


class AI_Network(nn.Module):
    def __init__(self):
        """
        Initializes the Agent class, which defines a simple neural network.
        """
        super().__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 20) # 중간 계층 추가
        self.fc3 = nn.Linear(20, 5)  # 최종 출력 계층
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU 계층
        
    def forward(self, s):
        s1 = self.leaky_relu(self.fc1(s))  # Leaky ReLU 적용
        s2 = self.leaky_relu(self.fc2(s1))  # 중간 계층에 Leaky ReLU 적용
        s3 = self.fc3(s2)  # 최종 선형 계층

        output = softmax(s3, dim=1)  # Softmax 활성화 함수로 확률 출력
        print(f's1 {s1}| s2 {s2}| output {output}')
        return output
    
    def save_network(self, path='AINetwork.pt'):
        torch.save(self.state_dict(), path)

    def load_network(self, path='AINetwork.pt'):
        self.load_state_dict(torch.load(path))

if __name__ == "__main__":

    ## inputs

    # 원본 네트워크 가중치 출력
    Net_orgin = AI_Network()
    print(Net_orgin.fc1.weight.data)

    # 네트워크 저장
    Net_orgin.save_network()
    print('Save orgin_network Done ==================== ')

    # 새로운 네트워크 생성 후
    Net_new = AI_Network()
    print('New_network ==================== ')
    # 가중치 출력
    print(Net_new.fc1.weight.data)

    # 저장된 가중치 로드
    Net_new.load_network('AINetwork.pt')
    print('Load New_network ==================== ')
    print(Net_new.fc1.weight.data)