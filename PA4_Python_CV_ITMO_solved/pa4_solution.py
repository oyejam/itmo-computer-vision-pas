"""Solved code for PA4: Viola-Jones face and body-part detection.

Run this file from the PA4_solved directory:
    python pa4_solution.py

The script saves annotated images to PA4_solved/outputs. The notebook version
uses ShowImages for inline visualization.
"""

from pathlib import Path
import cv2 as cv
import numpy as np

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_cascade(filename: str) -> cv.CascadeClassifier:
    cascade = cv.CascadeClassifier()
    path = str(BASE / filename)
    if not cascade.load(path):
        raise RuntimeError(f'Could not load cascade: {path}')
    return cascade


def read_image_or_frame(filename: str, frame_index: int | None = None) -> np.ndarray:
    path = str(BASE / filename)
    if frame_index is None:
        image = cv.imread(path, cv.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Could not read image: {path}')
        return image

    video = cv.VideoCapture(path)
    if not video.isOpened():
        raise RuntimeError(f'Could not open video: {path}')
    video.set(cv.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = video.read()
    video.release()
    if not ok or frame is None:
        raise RuntimeError(f'Could not read frame {frame_index} from {path}')
    return frame


def detect_faces(image: np.ndarray, cascade: cv.CascadeClassifier) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    return cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))


def save_face_detection_results() -> None:
    face_cascade = load_cascade('haarcascades/haarcascade_frontalface_default.xml')
    sources = [
        ('faces', 'images/faces.jpg', None),
        ('students_frame_35', 'images/video_students.mp4', 35),
        ('students_frame_95', 'images/video_students.mp4', 95),
    ]

    for name, filename, frame_index in sources:
        image = read_image_or_frame(filename, frame_index)
        faces = detect_faces(image, face_cascade)
        output = image.copy()
        for x, y, w, h in faces:
            cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv.imwrite(str(OUTPUT_DIR / f'{name}_faces.jpg'), output)
        print(f'{name}: {len(faces)} faces detected')


def save_part_detection_results() -> None:
    face_cascade = load_cascade('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = load_cascade('haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    smile_cascade = load_cascade('haarcascades/haarcascade_smile.xml')
    sources = [
        ('faces', 'images/faces.jpg', None),
        ('students_frame_35', 'images/video_students.mp4', 35),
        ('students_frame_95', 'images/video_students.mp4', 95),
    ]

    for name, filename, frame_index in sources:
        image = read_image_or_frame(filename, frame_index)
        faces = detect_faces(image, face_cascade)
        output = image.copy()
        total_eyes = 0
        total_mouths = 0

        for fx, fy, fw, fh in faces:
            cv.rectangle(output, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
            face_gray = cv.cvtColor(image[fy:fy + fh, fx:fx + fw], cv.COLOR_BGR2GRAY)
            face_gray = cv.equalizeHist(face_gray)

            eye_roi = face_gray[:fh * 2 // 3, :]
            eyes = eye_cascade.detectMultiScale(
                eye_roi,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(max(12, fw // 12), max(12, fh // 12)),
                maxSize=(max(20, fw // 2), max(20, fh // 3)),
            )
            total_eyes += len(eyes)
            for ex, ey, ew, eh in eyes:
                cv.rectangle(output, (fx + ex, fy + ey), (fx + ex + ew, fy + ey + eh), (0, 255, 0), 2)

            mouth_shift_y = fh * 2 // 3
            mouth_roi = face_gray[mouth_shift_y:, :]
            mouths = smile_cascade.detectMultiScale(
                mouth_roi,
                scaleFactor=1.05,
                minNeighbors=15,
                minSize=(max(16, fw // 6), max(10, fh // 12)),
                maxSize=(fw, max(20, fh // 2)),
            )
            total_mouths += len(mouths)
            for mx, my, mw, mh in mouths:
                cv.rectangle(
                    output,
                    (fx + mx, fy + mouth_shift_y + my),
                    (fx + mx + mw, fy + mouth_shift_y + my + mh),
                    (255, 0, 255),
                    2,
                )

        cv.imwrite(str(OUTPUT_DIR / f'{name}_parts.jpg'), output)
        print(f'{name}: {len(faces)} faces, {total_eyes} eyes, {total_mouths} mouths/smiles detected')


def process_video(display: bool = True) -> None:
    video = cv.VideoCapture(str(BASE / 'images/video_students.mp4'))
    if not video.isOpened():
        raise RuntimeError('Could not open images/video_students.mp4')

    face_cascade = load_cascade('haarcascades/haarcascade_frontalface_default.xml')
    last_frame = None
    frame_counter = 0

    if display:
        cv.namedWindow('Video face detection')

    while True:
        ok, frame = video.read()
        if not ok:
            break
        frame_counter += 1
        gray = cv.equalizeHist(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(24, 24))
        output = frame.copy()
        for x, y, w, h in faces:
            cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv.putText(output, f'Faces: {len(faces)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv.LINE_AA)
        last_frame = output
        if display:
            cv.imshow('Video face detection', output)
            if cv.waitKey(1) == 27 or cv.getWindowProperty('Video face detection', cv.WND_PROP_VISIBLE) < 1:
                break

    video.release()
    if display:
        cv.destroyAllWindows()
    if last_frame is not None:
        cv.imwrite(str(OUTPUT_DIR / 'video_last_frame_faces.jpg'), last_frame)
    print(f'Processed {frame_counter} video frames')


if __name__ == '__main__':
    save_face_detection_results()
    save_part_detection_results()
    # Set display=False if running headless and only the saved final frame is needed.
    process_video(display=False)
