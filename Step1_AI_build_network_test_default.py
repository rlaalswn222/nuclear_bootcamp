# 제일 기본 코드
# relu사용
# 계층 하나
import torch.nn as nn
import torch
from torch.nn.functional import softmax, relu
from torch.nn.functional import leaky_relu

class AI_Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the network structure with one fully connected layer
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 5) # Output size changed to 5 for classification

    def forward(self, s):
        """
        Forward pass of the network.
        """
        s1 = relu(self.fc1(s))
        s2 = self.fc2(s1)
        print(f's1 {s1}| s2 {s2}')
        return s2

if __name__ == "__main__":
    inputs = torch.tensor([[0.5, -0.2, 0.3], [0.1, 0.8, 0.1]], dtype=torch.float32)
    
    # Instantiate ExampleClass
    Net = AI_Network()
    Net.forward(inputs)