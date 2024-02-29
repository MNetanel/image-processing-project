from pathlib import Path
from platform import system

if system() == 'Linux':
    DATA_DIR = Path(r'/media/netanel/Backup/Image Processing Course Project/UBFC-RPPG')
else:
    DATA_DIR = Path(r'D:\Image Processing Course Project\UBFC-RPPG')

RAW_DATA_DIR = DATA_DIR / 'raw_data'
PREPROCESSED_DATA_DIR = DATA_DIR / 'preprocessed_data'
SUBJECTS = [1, 3, 4, 5, 8, 9, 10, 11, 12, 13]