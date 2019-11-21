import numpy as np
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def from_landmark_point(number: 'Union[int, Iterable]', landmarks) -> np.ndarray:
    if type(number) is int:
        ldm = landmarks.part(number)
        pnt = np.array([ldm.x, ldm.y], dtype=int)
        return pnt
    else:
        l = []
        for p in number:
            l.append(from_landmark_point(p, landmarks))
        return np.array(l, dtype=int)


def get_bounding(points: np.ndarray) -> np.ndarray:
    return np.array([points.min(axis=0), points.max(axis=0)])


def crop(frame: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return frame[start[1]:end[1], start[0]:end[0]]


def process_frame(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    #height, width, channels = frame.shape
    for _, face in enumerate(faces):
        landmarks = predictor(gray, face)

        # Face bounds image
        face_points = from_landmark_point(range(26 + 1), landmarks)
        start_point, end_point = get_bounding(face_points)
        cropped_face = crop(frame, start_point, end_point)
        return face_points, cropped_face

    else:
        pass
        #print('no faces')
        raise ReferenceError('Failed to detect faces')


def get_face(filename, resize=None):
    # try:
        _, f = process_frame(cv2.imread(filename))
        if resize:
            f = cv2.resize(f, resize, interpolation=cv2.INTER_CUBIC)
        return f
    # except:
    #     import shutil, os
    #     base_name = os.path.basename(filename)
    #     dest = os.path.join('probs', base_name)
    #     shutil.move(filename, dest)
    #     print(filename)
