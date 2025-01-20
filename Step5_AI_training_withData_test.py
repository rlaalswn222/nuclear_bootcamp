import torch.nn as nn
import torch
from torch.nn.functional import softmax, relu
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from Step4_AI_Dataloader_test import CustomDataset
from Step3_AI_training_test import AI_Network # 해당 파일에 있는 AI_network라는 class를 쓸 수 있음

Net_origin = AI_Network()

dataset = CustomDataset(file_path = './Day4_AI_Network/Data_minmax_labeling', time_seq =1) ### 변경
dataloader = DataLoader(dataset, batch_size=10, shuffle =True) ### 변경

for epoch in range(10): ### 변경
    for index, (sample_data, labels) in enumerate(dataloader):
        Net_origin.net_training(sample_data, labels)

    print(f"Epoch {epoch+1}")

Net_origin.save_network()

