import numpy as np
import dlib
import cv2
import h5py
from pathlib import Path
from src import RAW_DATA_DIR, PREPROCESSED_DATA_DIR

def avi2numpy(filepath: Path) -> np.ndarray:
    path = str(filepath)
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame.copy())
        else:
            break
    return np.array(frames)

def crop_face(frames: np.ndarray) -> np.ndarray:
    pass

def main():
    pass

if __name__ == '__main__':
    main()

