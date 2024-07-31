import pandas as pd
import os
import re

# Excel 파일 경로
excel_file = 'PBS_5_25.xlsx'

# 파일 이름에서 숫자 추출
numbers = re.findall(r'\d+', excel_file)
if len(numbers) >= 2:
    machine_count, job_count = map(int, numbers[-2:])
else:
    raise ValueError("파일 이름에서 machine 수와 job 수를 추출할 수 없습니다.")

# 결과를 저장할 디렉토리
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Excel 파일의 모든 시트 읽기
xlsx = pd.ExcelFile(excel_file)

for sheet_name in xlsx.sheet_names:
    # 시트 데이터 읽기
    df = pd.read_excel(xlsx, sheet_name)
    
    # 결과를 저장할 파일 경로
    base_name = os.path.splitext(excel_file)[0]  # 확장자 제거
    output_file = os.path.join(output_dir, f'{base_name}_{sheet_name}.txt')
    
    # 텍스트 파일 작성
    with open(output_file, 'w') as f:
        # 첫 줄에 job 수와 machine 수 작성
        f.write(f"{df.shape[0]}\t{df.shape[1]}\n")
        
        # 작업 시간 데이터 작성
        for _, row in df.iterrows():
            f.write('\t'.join(map(str, row)) + '\n')
        
        # machine 순서 작성 (모든 job에 대해 1부터 machine_count까지 순서)
        machine_order = '\t'.join(map(str, range(1, machine_count + 1)))
        for _ in range(df.shape[0]):
            f.write(f"{machine_order}\n")

print("모든 시트의 데이터가 텍스트 파일로 변환되었습니다.")