import cv2 as cv
import numpy as np
from typing import List, Tuple

def overlay_skeleton_nodes(img_gray: np.ndarray, skel: np.ndarray, nodes: List[Tuple[int, int]], endpoints: List[Tuple[int, int]]) -> np.ndarray:
    out = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    ys, xs = np.where(skel)
    out[ys, xs] = (0, 0, 255)  # Red for skeleton
    for node_id, (r, c) in enumerate(nodes, 1):
        cv.circle(out, (c, r), 5, (0, 255, 0), -1)  # Green for nodes, larger radius
        cv.putText(out, str(node_id), (c + 8, r - 8), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)  # Yellow text
    for idx, (r, c) in enumerate(endpoints):
        cv.circle(out, (c, r), 5, (255, 0, 0), -1)  # Blue for endpoints
        label = chr(ord('A') + idx)
        cv.putText(out, label, (c + 8, r - 8), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)  # Cyan text
    return out