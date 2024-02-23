import numpy as np
import dlib
import cv2
import h5py
from tqdm import tqdm
from pathlib import Path
from src import RAW_DATA_DIR, PREPROCESSED_DATA_DIR

def get_raw_files_list(data_dir) -> list[Path]:
    dirs = [d for d in data_dir.glob('*')]
    files = []
    for d in dirs:
        file = d / f'{d.name}_edited.avi'
        if not Path.exists(file):
            file = d / f'{d.name}-edited.avi'
        files.append(file)
    return files   

def avi2numpy(filepath: Path, crop_out_borders: bool = True, num_frames: int = 1800) -> np.ndarray:
    path = str(filepath)
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame.copy()[650:1000, 300:750, :] if crop_out_borders else frame.copy())
        else:
            raise Exception
    return np.array(frames)


def crop_face(frames: np.ndarray) -> np.ndarray:
    pass

def main():
    # videos to h5 file
    files = get_raw_files_list(RAW_DATA_DIR)
    cropped_frames_h5 = h5py.File(PREPROCESSED_DATA_DIR / 'cropped_frames.h5')
    for file in tqdm(files):
        file_numpy = avi2numpy(file)
        cropped_frames_h5.create_dataset(file.name[:-4], data=file_numpy, compression='gzip')
    
    
        

if __name__ == '__main__':
    main()

