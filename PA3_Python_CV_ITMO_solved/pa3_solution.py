import cv2 as cv
import numpy as np


MIN_MATCH_COUNT = 10
LOWE_RATIO = 0.75


def read_color(path):
    image = cv.imread(path, cv.IMREAD_COLOR)
    if not isinstance(image, np.ndarray) or image.data is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    return image


def detect_keypoints(image, detector_name='SIFT', nfeatures=500):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if detector_name.upper() == 'SIFT':
        detector = cv.SIFT_create(nfeatures=nfeatures)
    elif detector_name.upper() == 'ORB':
        detector = cv.ORB_create(nfeatures=nfeatures)
    else:
        raise ValueError('detector_name must be SIFT or ORB')
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def draw_features(image, detector_name='SIFT', nfeatures=100):
    keypoints, _ = detect_keypoints(image, detector_name, nfeatures)
    return cv.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def match_descriptors(desc1, desc2, detector_name='SIFT', matcher_name='BF_KNN', ratio=LOWE_RATIO):
    detector_name = detector_name.upper()
    matcher_name = matcher_name.upper()
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    if detector_name == 'SIFT':
        norm = cv.NORM_L2
        if matcher_name.startswith('FLANN'):
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            matcher = cv.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv.BFMatcher(norm, crossCheck=(matcher_name == 'BF_CC'))
    elif detector_name == 'ORB':
        norm = cv.NORM_HAMMING
        if matcher_name.startswith('FLANN'):
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            matcher = cv.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv.BFMatcher(norm, crossCheck=(matcher_name == 'BF_CC'))
    else:
        raise ValueError('detector_name must be SIFT or ORB')

    if matcher_name.endswith('KNN') or matcher_name == 'BF_KNN':
        raw_matches = matcher.knnMatch(desc1, desc2, k=2)
        matches = []
        for pair in raw_matches:
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                matches.append(pair[0])
    else:
        matches = matcher.match(desc1, desc2)

    return sorted(matches, key=lambda m: m.distance)


def find_object(query, scene, detector_name='SIFT', matcher_name='BF_KNN', min_matches=MIN_MATCH_COUNT):
    kp1, desc1 = detect_keypoints(query, detector_name, nfeatures=1500)
    kp2, desc2 = detect_keypoints(scene, detector_name, nfeatures=1500)
    matches = match_descriptors(desc1, desc2, detector_name, matcher_name)

    if len(matches) < min_matches:
        return None, None, kp1, kp2, matches, None, None

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    if H is None or mask is None:
        return None, None, kp1, kp2, matches, None, None

    h, w = query.shape[:2]
    box = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    projected_box = cv.perspectiveTransform(box, H)

    outlined_scene = scene.copy()
    cv.polylines(outlined_scene, [np.int32(projected_box)], True, (0, 0, 255), 3, cv.LINE_AA)
    result = cv.drawMatches(
        query,
        kp1,
        outlined_scene,
        kp2,
        matches,
        None,
        matchesMask=mask.ravel().tolist(),
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0),
    )
    return H, mask, kp1, kp2, matches, outlined_scene, result


def stitch_three(left, middle, right):
    def homography_to_middle(source, target):
        H, mask, *_ = find_object(source, target, 'SIFT', 'FLANN_KNN', min_matches=10)
        if H is None:
            raise RuntimeError('Could not estimate homography for panorama stitching.')
        if int(mask.sum()) < 10:
            raise RuntimeError('Too few inliers for reliable panorama stitching.')
        return H

    h, w = middle.shape[:2]
    canvas_h, canvas_w = h * 2, w * 3
    offset = np.array([[1, 0, w], [0, 1, h // 2], [0, 0, 1]], dtype=np.float64)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    y0, x0 = h // 2, w
    canvas[y0:y0 + h, x0:x0 + w] = middle

    for image in (left, right):
        H = offset @ homography_to_middle(image, middle)
        warped = cv.warpPerspective(image, H, (canvas_w, canvas_h))
        source_mask = np.any(warped > 0, axis=2)
        empty_mask = ~np.any(canvas > 0, axis=2)
        canvas[source_mask & empty_mask] = warped[source_mask & empty_mask]

    nonzero = np.argwhere(np.any(canvas > 0, axis=2))
    y_min, x_min = nonzero.min(axis=0)
    y_max, x_max = nonzero.max(axis=0) + 1
    return canvas[y_min:y_max, x_min:x_max]
