import os
import pandas as pd

# 경로 설정
folder_path = './test/demo'

# 폴더 내 파일 탐색
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        
        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # TICK 값이 750 초과인 행 제거
        df = df[df.iloc[:, 0] <= 750]

        # 변경된 데이터프레임 저장
        df.to_csv(file_path, index=False)

print("CSV 파일 처리가 완료되었습니다.")