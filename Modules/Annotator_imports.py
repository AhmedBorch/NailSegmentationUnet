import numpy as np
import cv2
import threading

MOVE = 0
SELECT = 1
ROTATE = 2
CROP = 3
SCALE = 4

GLOBAL_ANNOT = 0
GLOBAL_ORDER = 1

DATA_EMPTY = 0
DATA_REJECTED = -1
DATA_ANNOT = 1
DATA_APPROVED = 2

IMSIZEX = 192
IMSIZEY = 192

NOT_IN_IMG = (0, 0)


# Does P lie left of an infinite line defined by l1 and l2
# Code based on http://geomalgorithms.com/a03-_inclusion.html
def point_isleft(p, l1, l2):
    return (l1[0] - p[0]) * (l2[1] - p[1]) - (l2[0] - p[0]) * (l1[1] - p[1])


# Calculate the winding number of point P with respect to a polygon. P is inside poly if wn != 0
# Code based on http://geomalgorithms.com/a03-_inclusion.html
def poly_winding_num(poly, point):
    wn = 0
    curr_vertex = poly[0]
    for vertex in poly[1:]:
        if curr_vertex[1] <= point[1]:
            if vertex[1] > point[1] and point_isleft(point, curr_vertex, vertex) > 0:
                wn += 1
        else:
            if vertex[1] <= point[1] and point_isleft(point, curr_vertex, vertex) < 0:
                wn -= 1
        curr_vertex = vertex
    return wn


def inside_polygon(poly, point):
    return poly_winding_num(poly, point) != 0


def polygon_mask(mask_shape, landmark_set):
    mask = np.zeros(mask_shape, np.uint8)
    for landmarks in landmark_set:
        if len(landmarks) < 3:
            continue
        landmarks = landmarks.copy()
        landmarks.append(landmarks[0])
        xs = [x for (x, y) in landmarks]
        ys = [y for (x, y) in landmarks]
        bbxs, bbxe = np.min(xs), np.max(xs)  # Bounding box x
        bbys, bbye = np.min(ys), np.max(ys)  # Bounding box x
        for xi in range(bbxe - bbxs):
            for yi in range(bbye - bbys):
                mask[bbys + yi, bbxs + xi] |= int(inside_polygon(landmarks, (bbxs + xi, bbys + yi)))
    return mask