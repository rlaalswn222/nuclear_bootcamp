# LSTM으로 안 바꾸고
# leaky_relu사용
# softmax는 CrossEntropy에서 
import torch.nn as nn
import torch
from torch.nn.functional import softmax, leaky_relu
import torch.optim as optim

class AI_Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the network structure with one fully connected layer
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 5) 

    def forward(self, s):
        s1 = leaky_relu(self.fc1(s))
        s2 = leaky_relu(self.fc2(s1))
        s3 = self.fc3(s2)
        
        # softmax 함수는 CrossEntropyLoss에서 실행
       
        return s3
    
    def save_network(self, path = 'AINetwork.pt'):
        torch.save(self.state_dict(), path)

    def load_network(self, path='AINetwork.pt'):
        self.load_state_dict(torch.load(path))

    def net_training(self, training_data, training_target):
        self.opt = optim.Adam(self.parameters(), lr=0.001)
        self.cri = nn.CrossEntropyLoss() # 손실함수. nn.CrossEntropyLoss함수에서 정의된 손실 함수

        predict = self.forward(training_data)

        self.opt.zero_grad()

        loss = self.cri(predict, training_target) # (batcn, class=5) <=> (batch, index=1)

        loss.backward()

        self.opt.step()
        
    def calculate_accuracy(self, network, data, target):
        with torch.no_grad():
            predictions = network(data)
            predicted_classes = torch.argmax(predictions, dim=1)
            correct_predictions = (predicted_classes == target).sum().item()
            accuracy = (correct_predictions / len(target))*100
        return accuracy


if __name__ == "__main__":
    Net_orgin = AI_Network()

    for epoch in range(10):
        inputs = torch.tensor([[0.5, -0.2, 0.3], [0.1, 0.8, 0.1], [0.2, 0.2, 0.4], [0.2, 0.1, 0.7], [-0.2, 0.8, 0.6]], dtype=torch.float32)
        targets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long) # Batch 3

        Net_orgin.net_training(inputs, targets)
        training_loss = Net_orgin.cri(Net_orgin(inputs), targets).item()
        accuracy = Net_orgin.calculate_accuracy(Net_orgin, inputs, targets)
        print(f"Epoch {epoch} | Loss: {training_loss:.4f} | Accuracy: {accuracy:.2f} %")

    Net_orgin.save_network()
