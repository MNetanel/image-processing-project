import h5py
import numpy as np
import cv2
import sys
import re
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path('.').absolute()))

from config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR

def vid2numpy(filepath: Path, y_start: int, y_end: int, x_start: int, x_end: int) -> np.ndarray:
    path = str(filepath)
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame.copy()[y_start:y_end, x_start:x_end, [2, 1, 0]])
        else:
            break
    return np.array(frames)

def main():
    # Get list of files
    dirs = [d for d in RAW_DATA_DIR.glob('*')]
    files = []
    for d in dirs:
        file = d / f'{d.name}_edited.avi'
        if not Path.exists(file):
            file = d / f'{d.name}-edited.avi'
        files.append(file)
    files = [file for file in files if 'M' not in file.name]
    
    # Convert each one to numpy
    y_start, y_end, x_start, x_end= (650, 1000, 300, 750)
    h5_file = h5py.File(str(PREPROCESSED_DATA_DIR / 'videos.h5'), 'w')
    for file in tqdm(files):
        condition_name = re.findall(r'.*(?=[_-]edited\.avi)', file.name)[0]
        frames = vid2numpy(file, y_start, y_end, x_start, x_end)
        h5_file.create_dataset(name=condition_name,
                                data=frames,
                                compression='gzip',
                                chunks=True)
    h5_file.close()    

if __name__ == '__main__':
    main()
