import sys
from functools import reduce
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import label, regionprops
from tqdm import tqdm


def vid2numpy(filepath: Path, y_start: int, y_end: int, x_start: int, x_end: int) -> np.ndarray:
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
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face = detector(gray)[0]
    landmarks = predictor(gray, face)
    return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(81)]


def _get_or_generate_frame_forehead_landmarks(frame_landmarks: list[tuple]):
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


def _get_kovac_mask_for_frame(frame):
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


def _get_largest_area_in_mask(mask):
    rps = regionprops(label(mask))
    coords = rps[np.argmax([r.area for r in rps])].coords.T
    out = np.zeros_like(mask)
    out[tuple(coords)] = 1
    return out.astype(np.uint8)


###################################################################
# FUNCTIONS THAT OPERATE ON A BATCH OF FRAMES

def get_forehead_landmarks_every_nth_frame(frames: np.ndarray, n: int, detector, predictor):
    landmarks = []
    for i in range(0, len(frames), n):
        frame_landmarks = _get_frame_landmarks(frames[i], detector, predictor)
        frame_forehead_landmarks = _get_or_generate_frame_forehead_landmarks(
            frame_landmarks)
        for _ in range(n):
            landmarks.append(frame_forehead_landmarks)
    return np.array(landmarks)


def crop_frames_by_landmarks(frames: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
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
    masks = []
    for frame in frames:
        kovac = _get_kovac_mask_for_frame(frame)
        mask = _get_largest_area_in_mask(kovac)
        masks.append(np.stack([mask] * 3, axis=-1))
    masks = np.array(masks)
    return frames * masks
