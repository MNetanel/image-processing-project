import h5py
import numpy as np
import cv2
import sys
import dlib
from pathlib import Path
from tqdm import tqdm
from functools import reduce
from skimage.measure import label, regionprops

src = Path(__file__).parent
sys.path.append(str(src))


def vid2numpy(filepath: Path, y_start: int, y_end: int, x_start: int, x_end: int) -> np.ndarray:
    '''
    Convert a video file to a numpy array of frames.

    Args:
        filepath (Path): The path to the video file.
        y_start (int): The starting y-coordinate for cropping the frames.
        y_end (int): The ending y-coordinate for cropping the frames.
        x_start (int): The starting x-coordinate for cropping the frames.
        x_end (int): The ending x-coordinate for cropping the frames.

    Returns:
        np.ndarray: The numpy array of frames.

    '''
    path = str(filepath)
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(
                frame.copy()[y_start:y_end, x_start:x_end, [2, 1, 0]])
        else:
            break
    return np.array(frames)


def _get_frame_landmarks(frame: np.ndarray, detector, predictor):
    '''
    Get the facial landmarks in `frame`.

    Args:
        frame (np.ndarray): The input frame.
        detector: The face detector.
        predictor: The landmark predictor.

    Returns:
        list: A list of tuples representing the (x, y) coordinates of the landmarks.
        None: If no face is detected in the frame.
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        face = detector(gray)[0]
        landmarks = predictor(gray, face)
        return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(81)]
    return None


def _get_or_generate_frame_forehead_landmarks(frame_landmarks: list[tuple]):
    '''
    Get or generate forehead landmarks based on the given frame facial landmarks.

    Args:
        frame_landmarks (list[tuple]): List of facial landmarks for the frame.

    Returns:
        list[tuple]: List of forehead landmarks.
    '''
    forehead_landmarks = []
    forehead_lst = [(79, 74), 79, 73, 72, 69, 76, (76, 75), (75, 74)]
    for lm in forehead_lst:
        if type(lm) == int:
            # If `lm` is an int, it is the index of a landmark
            forehead_landmarks.append(frame_landmarks[lm])
        elif type(lm) == tuple:
            # If `lm` is a tuple, generate a point that consists of the x of the first element and y of the second element.
            forehead_landmarks.append(
                (frame_landmarks[lm[0]][0], frame_landmarks[lm[1]][1]))
    return forehead_landmarks


def _get_kovac_mask_for_frame(frame: np.ndarray):
    '''
    Get a threshold-based segmentation mask for skin regions.
    Based on Kovac et al. (2003)

    Args:
        frame (np.ndarray): The frame on which to perform the masking.

    Returns:
        np.ndarray: The masked frame.
    '''
    R, G, B = frame.transpose(2, 0, 1)
    masks = [
        R > 95,
        G > 40,
        B > 20,
        frame.max(axis=2) - frame.min(axis=2) > 15,
        np.abs(R - G) > 15,
        R > G,
        R > B
    ]
    return reduce(np.bitwise_and, masks).astype(np.uint8)


def _get_largest_area_in_mask(mask: np.ndarray):
    '''
    Get the largest connected area in a binary mask (used to clean "noisy" masks).

    Parameters:
        mask (np.ndarray): Binary mask representing the areas of interest.

    Returns:
        np.ndarray: Binary mask with only the largest connected area.
    '''
    rps = regionprops(label(mask))
    coords = rps[np.argmax([r.area for r in rps])].coords.T
    out = np.zeros_like(mask)
    out[tuple(coords)] = 1
    return out.astype(np.uint8)


###################################################################
# FUNCTIONS THAT OPERATE ON A BATCH OF FRAMES

def get_forehead_landmarks_every_nth_frame(frames: np.ndarray, n: int, detector, predictor):
    '''
    Get or generate forehead landmarks on every `n`th frame.

    Args:
        frames (np.ndarray): The frames to get the landmarks of.
        n (int): Number of frames to skip.
        detector: Dace detector.
        predictor: Shape predictor.

    Returns:
        np.ndarray: An array containing the forehead landmarks.
    '''
    landmarks = []
    for i in range(0, len(frames), n):
        frame_landmarks = _get_frame_landmarks(frames[i], detector, predictor)
        if frame_landmarks:
            frame_forehead_landmarks = _get_or_generate_frame_forehead_landmarks(
                frame_landmarks)
        for _ in range(n):  # if not found, use from previous match
            landmarks.append(frame_forehead_landmarks)
    return np.array(landmarks)


def crop_frames_by_landmarks(frames: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    '''
    Crop frames based on landmarks.

    Args:
        frames (np.ndarray): Array of frames.
        landmarks (np.ndarray): Array of landmarks corresponding to each frame.

    Returns:
        np.ndarray: Array of cropped frames.

    '''
    masked_frames = []
    for i, frame in enumerate(frames):
        frame_mask = np.zeros_like(frame)
        frame_landmarks = np.array([landmarks[i]])
        cv2.fillPoly(frame_mask, frame_landmarks, [255] * 3)
        masked_frame = cv2.bitwise_and(frame, frame_mask)
        masked_frames.append(masked_frame.copy())

    masked_frames = np.array(masked_frames)

    x_min, x_max, y_min, y_max = (landmarks[..., 0].min(),
                                  landmarks[..., 0].max(),
                                  landmarks[..., 1].min(),
                                  landmarks[..., 1].max()
                                  )
    return masked_frames[:, y_min:y_max, x_min:x_max, :]


def segment_forehead(frames: np.ndarray):
    '''
    Segment the forehead region from `frames` using a threshold-based segmentation mask
    for skin regions. Based on Kovac et al. (2003).

    Args:
        frames (np.ndarray): The frames to extract the forehead area from.

    Returns:
        np.ndarray: Masked/segmented frames.
    '''
    masks = []
    for frame in frames:
        kovac = _get_kovac_mask_for_frame(frame)
        mask = _get_largest_area_in_mask(kovac)
        masks.append(np.stack([mask] * 3, axis=-1))
    masks = np.array(masks)
    return frames * masks

def preprocess_videos(raw_data_dir: Path, preprocessed_data_dir: Path):
    # 1. AVI files to h5 file
    files = [d / 'vid.avi' for d in raw_data_dir.glob('*')]
    
    # Convert each one to numpy
    y_start, y_end, x_start, x_end= (0, 400, 200, 500)
    h5_file = h5py.File(str(preprocessed_data_dir / 'videos_all.h5'), 'w')
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
    h5_in = h5py.File(preprocessed_data_dir / 'videos_all.h5')
    h5_out = h5py.File(preprocessed_data_dir / 'foreheads_all.h5', 'w')
    
    t = tqdm(h5_in.keys())
    for key in t:
        t.set_postfix_str(key)
        frames = np.array(h5_in[key])
        forehead_landmarks = get_forehead_landmarks_every_nth_frame(frames, 10, face_detector, landmark_predictor)
        cropped_frames = crop_frames_by_landmarks(frames, forehead_landmarks)
        forehead_frames = segment_forehead(cropped_frames)
        print(forehead_frames.shape)
        h5_out.create_dataset(name=key,
                              data=forehead_frames,
                              compression='gzip')

