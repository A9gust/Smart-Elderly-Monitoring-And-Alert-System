from demo import run_demo_, VideoReader
import cv2
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
import numpy as np
import torch
import random

class InMemoryVideoReader:
    def __init__(self, frames):
        self.frames = frames
        self.idx = 0
        self.max_idx = len(frames)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.max_idx:
            raise StopIteration
        frame = self.frames[self.idx]
        self.idx += 1
        return frame

def extract_keyframes(video_path, num_keyframes=50, seed=42):
    """
    Randomly extract keyframe indices and frames from a video.

    Args:
        video_path (str): Path to video file.
        num_keyframes (int): Number of keyframes to extract.
        seed (int): Random seed for reproducibility.

    Returns:
        List[int]: Indices of keyframes.
        List[np.ndarray]: The keyframe images themselves.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = list(range(total_frames))
    
    if total_frames == 0:
        return [], []

    random.seed(seed)
    keyframe_indices = sorted(random.sample(frame_indices, min(num_keyframes, total_frames)))

    keyframes = []
    for idx in keyframe_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            keyframes.append(frame)

    cap.release()
    print(len(keyframes))
    return keyframe_indices, keyframes

def static_pose_detection_from_keyframes(keyframes, fps=30, window_sec=1, movement_threshold=50.0):

    # Step 1: Load pose model
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('ckpts/checkpoint.pth', map_location='cpu')
    load_state(net, checkpoint)

    # Step 2: Run pose estimation only on keyframes
    image_provider = InMemoryVideoReader(keyframes)
    keypoints = run_demo_(net, image_provider, height_size=256, cpu=False, track=1, smooth=1)  # shape (T, 18, 2)
    # Step 3: Inline static pose detection (no helper function call)
    static_frame_indices = []
    window_size = fps * window_sec

    for t in range(window_size, len(keypoints)):
        current = keypoints[t]
        past = keypoints[t - window_size]

        displacement = np.linalg.norm(current - past, axis=1)  # (18,)
        avg_movement = np.mean(displacement)
        if avg_movement < movement_threshold:
            static_frame_indices.append(t)

    print("Static poses detected at keyframe indices:", static_frame_indices)
    return static_frame_indices
