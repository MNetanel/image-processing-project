RAW_DATA_DIR = r'your/raw/data/dir'
GT_DATA_DIR = r'your/ground_truth/data/dir'
PROCESSED_DATA_DIR = r'your/processed/data/dir'

with open('.env', 'w') as f:
    f.write(f'RAW_DATA_DIR={RAW_DATA_DIR}\n')
    f.write(f'PROCESSED_DATA_DIR={GT_DATA_DIR}\n')
    f.write(f'PROCESSED_DATA_DIR={PROCESSED_DATA_DIR}')
