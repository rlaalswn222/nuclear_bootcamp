# 변수 추리고 나서 진행하자

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    # time seq를 고려할때가 발생함 우선 default 값으로 time_seq를 1로 두자
    def __init__(self, file_path='./test/data', time_seq=1):
        #1. Data file path
        file_list = os.listdir(file_path)

        #2. Data append
        self.data = []
        self.label = []

        for file in file_list:
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(file_path, file))

                if time_seq > 1:
                    for i in range(len(df) - time_seq+1): # 단순히 데이터 길이 만큼이 아니라 time_seq만큼 빼준다
                        self.data.append(df.iloc[i:i + time_seq, :-1].values) # 뒤와 같이 단순히 append를 해주지만 [i부터 i+time_seq만큼]
                        self.label.append(df.iloc[i + time_seq - 1, -1]) # 뒤가 -1이면 맨 뒤 열
                

                    # Note
                    # LSTM을 사용할 경우에는 batch_first =True 해줘야함
                else:
                    for i in range(len(df)): # 행의 길이 만큼!
                        self.data.append(df.iloc[i, :-1].values) # 행의 정보를 갖고 오는 iloc 전체 길이 만큼 갖고 온다
                        self.label.append(df.iloc[i, -1]) # 맨 마지막 label(답안)따로 label에 대한 정보만  append

            
        # 3. list -> np.array (tensor로 바로 바꾸면 warning떠서 중간과정으로 np.array로 변경해줌)
        self.data = np.array(self.data)
        self.label = np.array(self.label)

        # 4. np.array -> torch.tensor
        self.data = torch.tensor(self.data, dtype = torch.float32)
        self.label = torch.tensor(self.label, dtype = torch.long)

        # result
        print(self.data.size())
        print(self.label.size())

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]

# data가 어떻게 돌아가고 있는지 main함수를 돌려봄   
if __name__ == "__main__":
    dataset = CustomDataset(file_path='./test/data', time_seq=4) # 맨앞에 생성한 def
    dataloader = DataLoader(dataset, batch_size=5, shuffle =True)

    # for index, (sample_data, labels) in enumerate(dataloader):
    #     print(f"{index}/{len(dataloader)}")
    #     print("x shape:", sample_data.size())
    #     print("y shape:", labels.size())
    
