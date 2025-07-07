import numpy as np
from scipy.ndimage import convolve, label, center_of_mass
from skimage.morphology import skeletonize
from collections import deque
from typing import List, Tuple, Dict

def skeletonise_image(bw: np.ndarray) -> np.ndarray:
    return skeletonize(bw).astype(np.uint8)

def neighbour_count(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return convolve(skel, kernel, mode="constant", cval=0) - skel

def find_endpoints(skel: np.ndarray) -> List[Tuple[int, int]]:
    deg = neighbour_count(skel)
    return [tuple(p) for p in np.argwhere((skel == 1) & (deg == 1))]

def find_nodes(skel: np.ndarray) -> List[Tuple[int, int]]:
    deg = neighbour_count(skel)
    node_mask = (skel == 1) & (deg >= 3)
    lbl, n_comp = label(node_mask)
    centroids = center_of_mass(node_mask, lbl, range(1, n_comp + 1))
    return [(int(round(r)), int(round(c))) for r, c in centroids]

def euclidean_distance(p: Tuple[int, int], q: Tuple[int, int]) -> float:
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))

def find_connections(skel: np.ndarray, nodes: List[Tuple[int, int]], max_distance: int = 100) -> Dict[int, List[int]]:
    node_pos_to_index = {pos: idx for idx, pos in enumerate(nodes)}
    connections = {i: [] for i in range(len(nodes))}
    h, w = skel.shape
    for i, (sr, sc) in enumerate(nodes):
        visited = set()
        queue = deque([(sr, sc, 0)])
        while queue:
            r, c, d = queue.popleft()
            if d > max_distance:
                continue
            if (r, c) in node_pos_to_index and node_pos_to_index[(r, c)] != i:
                j = node_pos_to_index[(r, c)]
                if j not in connections[i]:
                    connections[i].append(j)
                continue
            visited.add((r, c))
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and skel[nr, nc] and (nr, nc) not in visited:
                        queue.append((nr, nc, d + 1))
    return connections

def compute_distances_from_connections(nodes: List[Tuple[int, int]], connections: Dict[int, List[int]]) -> List[Tuple[int, int, int, int, int, int, float]]:
    result = []
    visited = set()
    for i, neighbors in connections.items():
        r1, c1 = nodes[i]
        for j in neighbors:
            if (i, j) in visited or (j, i) in visited:
                continue
            r2, c2 = nodes[j]
            d = euclidean_distance((r1, c1), (r2, c2))
            result.append((i + 1, r1, c1, j + 1, r2, c2, d))
            visited.add((i, j))
    return result
