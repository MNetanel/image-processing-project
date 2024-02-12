from pathlib import Path
from platform import system

if system() == 'Linux':
    DATA_DIR = Path(r'/media/netanel/Backup/Image Processing Course Project/data')
else:
    DATA_DIR = Path(r'D:\Image Processing Course Project\data')

RAW_DATA_DIR = DATA_DIR / 'raw'
PREPROCESSED_DATA_DIR = DATA_DIR / 'preprocessed'