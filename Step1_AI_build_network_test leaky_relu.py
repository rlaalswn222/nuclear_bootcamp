# 활성화함수 leaky_relu
# softmax 활성화함수 추가
# 중간계층 추가 (fc2, s2)
# sigmoid 

import torch.nn as nn
import torch
from torch.nn.functional import softmax
from torch.nn.functional import leaky_relu

class AI_Network(nn.Module):
    def __init__(self):
        """
        Initializes the Agent class, which defines a simple neural network.
        """
        super().__init__()
        # Define the network structure with one fully connected layer
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 20) ## 중간계층 추가 # 비선형성을 높임으로 
        self.fc3 = nn.Linear(20, 5) # Output size changed to 5 for classification
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  ## Leaky ReLU 계층 추가


    def forward(self, s):
        s1 = leaky_relu(self.fc1(s))
        s2 = leaky_relu(self.fc2(s1)) ## 중간계층 추가함에 따라 신경망 구조 추가
        s3 = self.fc3(s2)
        output = softmax(s3, dim=1)  # Softmax 활성화 함수 추가
        print(f's1 {s1}| s2 {s2}| output {output}')
        return output

if __name__ == "__main__":
    inputs = torch.tensor([[0.5, -0.2, 0.3], [0.1, 0.8, 0.1]], dtype=torch.float32)
    
    # Instantiate ExampleClass
    Net = AI_Network()
    Net.forward(inputs)