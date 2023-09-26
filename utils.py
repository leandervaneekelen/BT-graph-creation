from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
from collections import defaultdict
import numpy as np


def parse_xml(file_path) -> dict:
    wsa = WholeSlideAnnotation(annotation_path=file_path)
    annos_per_label = wsa.annotations_per_label
    coords_per_label = defaultdict(list)
    for label, annos in annos_per_label.items():
        for anno in annos:
            if anno.type == "polygon":
                coords = tuple(map(tuple, anno.coordinates))
            elif anno.type == "point":
                coords = anno.coordinates.squeeze()
            else:
                raise NotImplementedError
            coords_per_label[label].append(coords)
    return coords_per_label


def make_closed_loop(x1, y1, x2, y2):
    """
    Given a bounding box (x1, y1, x2, y2), return a closed loop of coordinates
    describing the polygon.
    """
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])


def xywh2xyxy(xywh):
    """
    Convert bounding box from xywh to xyxy format.
    """
    x, y, w, h = xywh
    return np.array([x, y, x + w, y + h])
