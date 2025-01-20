import pandas as pd
import os

# Min-Max Scaling 함수 선언
def minmax(y, y_min, y_max):
    if y_min == y_max:  # 최소값과 최대값이 같을 경우
        return 0  # 모든 값을 0으로 설정
    else:
        return (y - y_min) / (y_max - y_min)

# 폴더 경로 설정
folder_path = './code/record_test'

# 모든 파일을 읽어서 하나의 데이터 프레임으로 합치기
all_data = pd.DataFrame()

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    print(f"Processing file: {file_path}")
    db = pd.read_csv(file_path)
    all_data = pd.concat([all_data, db], ignore_index=True)

# 숫자형 열만 선택
numeric_columns = all_data.select_dtypes(include=['number']).columns

# 각 열의 min, max 값 구하기
para_min = {col: all_data[col].min() for col in numeric_columns}
para_max = {col: all_data[col].max() for col in numeric_columns}
print("Parameter Min Values:", para_min)
print("Parameter Max Values:", para_max)

# 결과 저장을 위해 minmax.csv 파일 생성 (min, max 값을 저장)
with open('minmax_scaled_data.csv', 'w') as f:
    f.write(','.join(numeric_columns) + '\n')
    f.write(','.join(str(para_min[col]) for col in numeric_columns) + '\n')
    f.write(','.join(str(para_max[col]) for col in numeric_columns) + '\n')

# 숫자형 열에 Min-Max Scaling 적용
data_scaled = all_data.copy()

for col in numeric_columns:
    data_scaled[col] = data_scaled[col].apply(
        lambda x: minmax(x, para_min[col], para_max[col])
    )

# 스케일링된 데이터 저장
output_path = './code/minmax_scaled_data.csv'
data_scaled.to_csv(output_path, index=False)
print(f"Min-Max Scaled data saved to {output_path}")
