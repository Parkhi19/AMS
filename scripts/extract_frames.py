import cv2
import os
import face_recognition
from frame_save_duration import save_frames_duration
import sys

def get_frames_from_video(video_file):
    save_frames_per_second = 10
    frame_number = 0
    filename, _ = os.path.splitext(video_file)
    if not os.path.isdir(filename):
        os.mkdir(filename)

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if video_fps == 0 or frame_count == 0:
        print("Error: FPS or frame count is zero.")
        return

    save_frames_per_second = min(video_fps, save_frames_per_second)
    frames_duration_to_save = save_frames_duration(cap, save_frames_per_second)

    frames_processed = 0
    while True:
        read_current_frame, frame = cap.read()
        if not read_current_frame:
            break

        frame_duration = frames_processed / video_fps
        closest_duration = frames_duration_to_save[0]

        if frame_duration >= closest_duration:
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)

            for top, right, bottom, left in face_locations:
                face_image = frame[top:bottom, left:right]
                cv2.imwrite(f"face_image_{frame_number}.jpg", face_image)
                frame_number += 1
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.imshow('Video', frame)

            if cv2.waitKey(10) == 27:
                break

            try:
                frames_duration_to_save.pop(0)
            except IndexError:
                pass

        frames_processed += 1

    cap.release()
    cv2.destroyAllWindows()
