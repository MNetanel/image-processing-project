import h5py
import numpy as np
import cv2
import sys
import dlib
from pathlib import Path
from tqdm import tqdm
from preprocessing_utils import vid2numpy, get_forehead_landmarks_every_nth_frame,crop_frames_by_landmarks, segment_forehead
src = Path(__file__).parents[1]
sys.path.append(str(src))
from config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR, SUBJECTS



def main():
    # 1. AVI files to h5 file
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
    
    
    # 2. segment foreheads in h5 file
    predictor_path = str(src / 'external_utils/shape_predictor_81_face_landmarks.dat')
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(predictor_path)
    h5_in = h5py.File(PREPROCESSED_DATA_DIR / 'videos.h5')
    h5_out = h5py.File(PREPROCESSED_DATA_DIR / 'foreheads.h5', 'w')
    
    t = tqdm(h5_in.keys())
    for key in t:
        t.set_postfix_str(key)
        frames = np.array(h5_in[key])
        forehead_landmarks = get_forehead_landmarks_every_nth_frame(frames, 10, face_detector, landmark_predictor)
        cropped_frames = crop_frames_by_landmarks(frames, forehead_landmarks)
        forehead_frames = segment_forehead(cropped_frames)
        h5_out.create_dataset(name=key,
                              data=forehead_frames,
                              compression='gzip')

if __name__ == '__main__':
    main()