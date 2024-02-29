import h5py
import numpy as np
import cv2
import sys
import re
from pathlib import Path
from tqdm import tqdm
src = Path(__file__).parents[1]
sys.path.append(str(src))

from config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR, SUBJECTS

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
    files = [RAW_DATA_DIR / f'subject{subject}/vid.avi' for subject in SUBJECTS]
    
    # Convert each one to numpy
    y_start, y_end, x_start, x_end= (0, 400, 200, 500)
    h5_file = h5py.File(str(PREPROCESSED_DATA_DIR / 'videos.h5'), 'w')
    t = tqdm(files)
    for file in t:
        name = file.parent.name # Name of directory
        t.set_postfix_str(name)
        frames = vid2numpy(file, y_start, y_end, x_start, x_end)
        h5_file.create_dataset(name=name,
                                data=frames,
                                compression='gzip',
                                chunks=True)
    h5_file.close()    

if __name__ == '__main__':
    main()
