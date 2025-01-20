import torch.nn as nn
import torch
from torch.nn.functional import softmax, relu


class AI_Network(nn.Module):
    def __init__(self):
        """
        Initializes the Agent class, which defines a simple neural network.
        """
        super().__init__()
        # Define the network structure with one fully connected layer
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 5) # Output size changed to 5 for classification

    def forward(self, s):
        s1 = relu(self.fc1(s))
        s2 = self.fc2(s1)
        
        print(f's1 {s1}| s2 {s2} ')
        return s2
    
    def save_network(self):
        torch.save(self.state_dict(), 'AINetwork.pt')

    def load_network(self, path='AINetwork.pt'):
        self.load_state_dict(torch.load(path))

if __name__ == "__main__":
    Net_orgin = AI_Network()

    print(Net_orgin.fc1.weight.data)

    Net_orgin.save_network()

    print('Save orgin_network Done ==================== ')

    Net_new = AI_Network()

    print('New_network ==================== ')
    print(Net_new.fc1.weight.data)

    Net_new.load_network('AINetwork.pt')

    print('Load New_network ==================== ')
    print(Net_new.fc1.weight.data)