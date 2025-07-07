import cv2 as cv
import numpy as np
from typing import List, Tuple

def overlay_skeleton_nodes(img_gray: np.ndarray, skel: np.ndarray, nodes: List[Tuple[int, int]], endpoints: List[Tuple[int, int]]) -> np.ndarray:
    out = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    ys, xs = np.where(skel)
    out[ys, xs] = (0, 0, 255)
    for node_id, (r, c) in enumerate(nodes, 1):
        cv.circle(out, (c, r), 3, (0, 255, 0), -1)
        cv.putText(out, str(node_id), (c + 6, r - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for idx, (r, c) in enumerate(endpoints):
        cv.circle(out, (c, r), 3, (255, 0, 0), -1)
        label = f"e{idx}"
        cv.putText(out, label, (c + 6, r - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return out
