import torch.nn as nn
import torch
from torch.nn.functional import softmax, relu
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from Step4_AI_Dataloader_test import CustomDataset
from Step3_AI_training_test_leaky_relu import AI_Network
from CNSenv import CommunicatorCNS
from collections import deque

import csv 

class SequenceBuffer:
    def __init__(self, sequence_length):
        """
        시퀀스를 원하는 길이만큼 모아서 반환하는 버퍼 클래스
        :param sequence_length: AI 모델에 입력으로 보낼 시퀀스 길이
        """
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=sequence_length)  # 최대 길이를 제한하여 오래된 데이터 자동 삭제

    def add_data(self, new_data):
        """
        새로운 데이터를 추가하고, 시퀀스가 충분하면 반환
        :param new_data: 새로운 입력 데이터 리스트
        :return: 시퀀스 길이가 충분하면 리스트 반환, 부족하면 None 반환
        """
        self.buffer.append(new_data)  # 데이터 추가
        
        if len(self.buffer) == self.sequence_length:  # 시퀀스 길이가 충족되면 반환
            return list(self.buffer)  # deque → list 변환 후 반환
        
        return None  # 아직 데이터가 부족하면 None 반환


def minmax(y, y_min, y_max):
    """Min-Max 정규화 함수"""
    if y_min == y_max:
        return 0
    else:
        return (y - y_min) / (y_max - y_min)
    

if __name__ == "__main__":
    # 모델 설정
    time_seq = 15
    input_size = 220
    hidden_size = 128
    num_layer = 2
    output_size = 5
    batch_size = 64
    sequence_length = 15

    # 버퍼 초기화
    buffer = SequenceBuffer(sequence_length)

    # AI 네트워크 로드
    Net_orgin = AI_Network(input_size, hidden_size, num_layer, output_size)
    Net_orgin.load_network('./AINetwork.pt') # 저장된 모델 가져오기

    # CNS 환경 설정
    cnsenv = CommunicatorCNS(com_ip= '192.168.0.8', com_port=7132)

    # Min-Max 정규화 파라미터 로드
    with open('./minmax_scaled_data.csv','r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = list(reader)

        para_list = lines[0]      
        para_min = [float(x) for x in lines[1]]
        para_max = [float(x) for x in lines[2]]

    print(f"데이터가 {sequence_length}개 모일 때까지 대기 중...")

    #=============================================================
    while True:
        is_updated = cnsenv.read_data()

        if is_updated:
            db_line = []
            for j, para in enumerate(para_list):
                db_line.append(minmax(cnsenv.mem[para]['Val'], para_min[j], para_max[j]))

            # 버퍼에 데이터 추가
            sequence = buffer.add_data(db_line)

            if sequence:
                print(f"데이터 {sequence_length}개가 쌓였음! AI 모델에 전달할 시퀀스:")

                # Torch Tensor 변환 (batch 차원 추가)
                inputs = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # (1, seq_length, input_size)

                # 모델 예측
                with torch.no_grad():  # 그래디언트 계산 방지
                    out = Net_orgin.forward(inputs)

                # Softmax 적용 후 argmax로 가장 확률 높은 클래스 선택
                out = torch.argmax(softmax(out, dim=1)).item()

                print(f'Net Predict : {out}')

            else:
                remaining = sequence_length - len(buffer.buffer)  # 남은 데이터 개수 확인
                print(f"데이터가 부족함. {remaining}개 더 필요함...")