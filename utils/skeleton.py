import cv2 as cv
import numpy as np
import csv
from .graph_analysis import (
    skeletonise_image, find_nodes, find_endpoints,
    find_connections, compute_distances_from_connections,
    euclidean_distance
)
from .visualization import overlay_skeleton_nodes

def load_binary(path: str) -> np.ndarray:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    _, bw = cv.threshold(img, 127, 1, cv.THRESH_BINARY)
    return bw.astype(np.uint8)

def Core_code(imput_image, output_csv, output_image):
    bw = load_binary(imput_image)
    skel = skeletonise_image(bw)
    img_gray = cv.imread(imput_image, cv.IMREAD_GRAYSCALE)
    nodes = find_nodes(skel)
    endpoints = find_endpoints(skel)

    connections = find_connections(skel, nodes)
    distances = compute_distances_from_connections(nodes, connections)

    endpoint_distances = []
    used_node_ids = set()
    for idx, (er, ec) in enumerate(endpoints):
        min_dist, nearest = float("inf"), -1
        for i, (nr, nc) in enumerate(nodes):
            d = euclidean_distance((er, ec), (nr, nc))
            if d < min_dist:
                min_dist, nearest = d, i
        if nearest != -1:
            endpoint_distances.append((nearest + 1, nodes[nearest], f"e{idx}", (er, ec), min_dist))
            used_node_ids.add(nearest + 1)

    connected_node_ids = {i for i, *_ in distances}
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Node1_ID', 'Coord1', 'Node2_ID', 'Coord2', 'Distance', 'End_point', 'E_Coord', 'E_Distance'])
        for (i, r1, c1, j, r2, c2, d) in distances:
            match = next((ep for ep in endpoint_distances if ep[0] == i), None)
            if match:
                _, _, label, (er, ec), ed = match
                w.writerow([i, f"({r1},{c1})", j, f"({r2},{c2})", f"{d:.5f}", label, f"({er},{ec})", f"{ed:.5f}"])
            else:
                w.writerow([i, f"({r1},{c1})", j, f"({r2},{c2})", f"{d:.5f}", 'null', 'null', 'null'])
        for i, (r, c) in enumerate(nodes, 1):
            if i not in connected_node_ids:
                match = next((ep for ep in endpoint_distances if ep[0] == i), None)
                if match:
                    _, _, label, (er, ec), ed = match
                    w.writerow([i, f"({r},{c})", 'null', 'null', 'null', label, f"({er},{ec})", f"{ed:.5f}"])

    out_img = overlay_skeleton_nodes(img_gray, skel, nodes, endpoints)
    cv.imwrite(output_image, out_img)

    return out_img

