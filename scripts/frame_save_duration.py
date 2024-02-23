import cv2
import numpy as np
import os

def save_frames_duration(cap, required_fps):
    frames_duration = []
    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)
    for current_frame_duration in np.arange(0, video_duration, 1/required_fps):
        frames_duration.append(current_frame_duration)
    return frames_duration        