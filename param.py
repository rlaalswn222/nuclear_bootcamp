import os
import pandas as pd

# THINGSTOBECONSIDERED.py 파일에서 final_param 읽기
code_path = "./code/THINGSTOBECONSIDERED.py"
with open(code_path, 'r', encoding='utf-8') as f:
    content = f.read()

# final_param 추출
import re
final_param = re.search(r'final_param\s*=\s*\[(.*?)\]', content, re.S)
if final_param:
    columns_to_keep = [col.strip().strip("'\"") for col in final_param.group(1).split(',')]
else:
    raise ValueError("final_param이 THINGSTOBECONSIDERED.py에서 발견되지 않았습니다.")

# CSV 파일 처리
input_folder = "./code/record_test"
output_folder = "./code/processed_data"
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.startswith("final") and file_name.endswith(".csv"):
        file_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # 필요한 열만 추출
        filtered_df = df[columns_to_keep]

        # 전처리된 파일 저장
        filtered_df.to_csv(output_path, index=False)

print(f"모든 파일이 {output_folder} 폴더에 저장되었습니다.")
