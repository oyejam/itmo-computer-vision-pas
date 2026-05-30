"""Complete PA4 solution: Viola-Jones face and face-part detection.

Run from the repository root or from this directory:
    python3 PA4_Python_CV_ITMO_solved/pa4_solution.py

The script writes annotated images to PA4_Python_CV_ITMO_solved/outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Source:
    title: str
    filename: str
    frame_index: int | None = None

    @property
    def output_stem(self) -> str:
        stem = Path(self.filename).stem
        if self.frame_index is None:
            return stem
        return f"{stem}_frame_{self.frame_index:03d}"


SOURCES = [
    Source("Group photo: many small faces", "images/faces.jpg"),
    Source("Movie scene: medium and large faces", "images/pa4_selfwork_group_scene.png"),
    Source("Large portrait: close face scale", "images/face.png"),
]


def load_cascade(filename: str) -> cv.CascadeClassifier:
    path = BASE / filename
    cascade = cv.CascadeClassifier()
    if not cascade.load(str(path)):
        raise RuntimeError(f"Could not load cascade: {path}")
    return cascade


def read_image_or_video_frame(source: Source) -> np.ndarray:
    path = BASE / source.filename
    if source.frame_index is None:
        image = cv.imread(str(path), cv.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        return image

    video = cv.VideoCapture(str(path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    video.set(cv.CAP_PROP_POS_FRAMES, source.frame_index)
    ok, frame = video.read()
    video.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {source.frame_index} from {path}")
    return frame


def prepare_gray(image: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return cv.equalizeHist(gray)


def detect_faces(
    image: np.ndarray,
    face_cascade: cv.CascadeClassifier,
    *,
    scale_factor: float = 1.08,
    min_neighbors: int = 8,
    min_size: tuple[int, int] = (24, 24),
) -> np.ndarray:
    return face_cascade.detectMultiScale(
        prepare_gray(image),
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    cv.putText(image, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(image, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv.LINE_AA)


def annotate_faces(image: np.ndarray, faces: np.ndarray) -> np.ndarray:
    output = image.copy()
    for index, (x, y, w, h) in enumerate(faces, start=1):
        cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
        draw_label(output, str(index), (x, max(18, y - 6)))
    return output


def detect_face_parts(
    image: np.ndarray,
    faces: np.ndarray,
    eye_cascade: cv.CascadeClassifier,
    smile_cascade: cv.CascadeClassifier,
) -> tuple[np.ndarray, int, int]:
    output = image.copy()
    total_eyes = 0
    total_mouths = 0

    for face_index, (fx, fy, fw, fh) in enumerate(faces, start=1):
        cv.rectangle(output, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
        draw_label(output, f"F{face_index}", (fx, max(18, fy - 6)))

        face_gray = prepare_gray(image[fy : fy + fh, fx : fx + fw])

        eye_roi_h = fh * 2 // 3
        eye_roi = face_gray[:eye_roi_h, :]
        eyes = eye_cascade.detectMultiScale(
            eye_roi,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(max(8, fw // 16), max(8, fh // 16)),
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
            minSize=(max(12, fw // 6), max(6, fh // 16)),
            maxSize=(fw, max(20, fh // 2)),
        )

        filtered_mouths = []
        for mx, my, mw, mh in mouths:
            mouth_center_x = mx + mw / 2
            if fw * 0.15 <= mouth_center_x <= fw * 0.85:
                filtered_mouths.append((mx, my, mw, mh))

        total_mouths += len(filtered_mouths)
        for mx, my, mw, mh in filtered_mouths:
            cv.rectangle(
                output,
                (fx + mx, fy + mouth_shift_y + my),
                (fx + mx + mw, fy + mouth_shift_y + my + mh),
                (255, 0, 255),
                2,
            )

    return output, total_eyes, total_mouths


def save_face_detection_results() -> list[tuple[Source, np.ndarray, np.ndarray]]:
    face_cascade = load_cascade("haarcascades/haarcascade_frontalface_default.xml")
    results = []

    for source in SOURCES:
        image = read_image_or_video_frame(source)
        faces = detect_faces(image, face_cascade)
        output = annotate_faces(image, faces)
        cv.imwrite(str(OUTPUT_DIR / f"{source.output_stem}_faces.jpg"), output)
        print(f"{source.title}: {len(faces)} faces detected")
        results.append((source, image, faces))

    return results


def save_part_detection_results(results: list[tuple[Source, np.ndarray, np.ndarray]]) -> None:
    eye_cascade = load_cascade("haarcascades/haarcascade_eye.xml")
    smile_cascade = load_cascade("haarcascades/haarcascade_smile.xml")

    for source, image, faces in results:
        output, total_eyes, total_mouths = detect_face_parts(image, faces, eye_cascade, smile_cascade)
        cv.imwrite(str(OUTPUT_DIR / f"{source.output_stem}_parts.jpg"), output)
        print(f"{source.title}: {len(faces)} faces, {total_eyes} eyes, {total_mouths} mouths/smiles detected")


def process_video(max_frames: int | None = None, display: bool = False) -> None:
    video_path = BASE / "images/pa4_selfwork_video.mp4"
    video = cv.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    face_cascade = load_cascade("haarcascades/haarcascade_frontalface_default.xml")
    last_frame = None
    frame_counter = 0

    if display:
        cv.namedWindow("Video face detection")

    while True:
        ok, frame = video.read()
        if not ok:
            break
        frame_counter += 1

        faces = detect_faces(frame, face_cascade, scale_factor=1.15, min_neighbors=8)
        output = annotate_faces(frame, faces)
        cv.putText(output, f"Faces: {len(faces)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv.LINE_AA)
        last_frame = output

        if display:
            cv.imshow("Video face detection", output)
            if cv.waitKey(1) == 27 or cv.getWindowProperty("Video face detection", cv.WND_PROP_VISIBLE) < 1:
                break

        if max_frames is not None and frame_counter >= max_frames:
            break

    video.release()
    if display:
        cv.destroyAllWindows()

    if last_frame is not None:
        cv.imwrite(str(OUTPUT_DIR / "video_last_frame_faces.jpg"), last_frame)
    print(f"Processed {frame_counter} video frames")


def main() -> None:
    results = save_face_detection_results()
    save_part_detection_results(results)
    process_video(display=False)


if __name__ == "__main__":
    main()
