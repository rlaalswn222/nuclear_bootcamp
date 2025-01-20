import os
import pandas as pd
import numpy as np

# 데이터 폴더 경로
file_path = './code/record_test'

# 폴더 내 파일 리스트 가져오기
file_list = [f for f in os.listdir(file_path) if f.endswith('.csv')]

for file_name in file_list:
    # 파일 경로 생성
    full_path = os.path.join(file_path, file_name)

    # CSV 파일 읽기
    data = pd.read_csv(full_path)

    # 파일명이 'loca'로 시작하는 경우 = 1
    if file_name.startswith('loca'):
        data['label'] = np.where(data.index < 150, 0, 1)
    
    # SGTR = 2
    elif file_name.startswith('SGTR'):
        data['label'] = np.where(data.index < 150, 0, 2)
    
    # MSLB out = 4
    elif file_name.startswith('MSLB__out'):
        data['label'] = np.where(data.index < 150, 0, 4)
    
    # MSLB in = 3
    else:
        data['label'] = np.where(data.index < 150, 0, 3)

    # 수정된 데이터 저장 (덮어쓰기)
    save_path = os.path.join(file_path, f'final_{file_name}')
    data.to_csv(save_path, index=False)
    print(f"Processed and saved: {save_path}")
