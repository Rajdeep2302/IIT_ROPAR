import os
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, label
from .graph_analysis import *


def load_binary(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image from {img_path}")
    _, bw = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return bw.astype(np.uint8), img


def angle_between(p1, p2):
    dy = p2[0] - p1[0]
    dx = p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx)) % 180


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def fit_chain_ellipses(region_nodes, region_ends, out_img, ellipse_data, id_start):
    count = id_start
    visited = set()

    connections = {}
    all_points = region_nodes + region_ends

    for p1 in all_points:
        if p1 in visited:
            continue
        # Find nearest unvisited neighbor within same region
        candidates = [p2 for p2 in all_points if p2 != p1 and (p1, p2) not in visited and (p2, p1) not in visited]
        if not candidates:
            continue

        p2 = min(candidates, key=lambda x: euclidean_dist(p1, x))

        mid = midpoint(p1, p2)
        major = euclidean_dist(p1, p2) / 2
        minor = max(3, major * 0.5)
        angle = angle_between(p1, p2)

        ellipse_data.append({
            'pair_id': count,
            'x': mid[1],
            'y': mid[0],
            'semi_major': major,
            'semi_minor': minor,
            'angle': angle
        })

        cv2.ellipse(out_img, (mid[1], mid[0]), (int(major), int(minor)),
                    angle, 0, 360, (0, 255, 0), 2)
        # cv2.putText(out_img, str(count), (mid[1] + 5, mid[0] - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        visited.add(p1)
        visited.add(p2)
        count += 1

    return count


def eclipse_image(img_path, out_img_path, out_csv_path):
    bw, img_gray = load_binary(img_path)
    skel = skeletonize(bw).astype(np.uint8)
    nodes = find_nodes(skel)
    endpoints = find_endpoints(skel)

    print(f"  Found {len(nodes)} nodes and {len(endpoints)} endpoints")

    out_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    ys, xs = np.where(skel)
    out_img[ys, xs] = (0, 0, 255)

    for idx, (r, c) in enumerate(nodes):
        cv2.circle(out_img, (c, r), 3, (0, 255, 0), -1)
        # cv2.putText(out_img, str(idx + 1), (c + 5, r - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    for idx, (r, c) in enumerate(endpoints):
        cv2.circle(out_img, (c, r), 3, (255, 0, 0), -1)
        # cv2.putText(out_img, f"e{idx}", (c + 5, r - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    labeled, num = label(bw)
    ellipse_data = []
    count = 1

    for region in range(1, num + 1):
        mask = labeled == region
        region_nodes = [p for p in nodes if mask[p]]
        region_ends = [p for p in endpoints if mask[p]]

        # Draw ellipses between endpoint-node and node-node
        count = fit_chain_ellipses(region_nodes, region_ends, out_img, ellipse_data, count)

    total_possible_pairs = 0

    for region in range(1, num + 1):
        mask = labeled == region
        region_nodes = [p for p in nodes if mask[p]]
        region_ends = [p for p in endpoints if mask[p]]
        region_all = region_nodes + region_ends

        # Count unique node-node and endpoint-node pairs
        unique_pairs = set()

        # Node-node pairs
        for i in range(len(region_nodes)):
            for j in range(i + 1, len(region_nodes)):
                unique_pairs.add(tuple(sorted((region_nodes[i], region_nodes[j]))))

        # Endpoint-node pairs (closest only)
        for ept in region_ends:
            if not region_nodes:
                continue
            closest_node = min(region_nodes, key=lambda n: euclidean_dist(ept, n))
            unique_pairs.add(tuple(sorted((ept, closest_node))))

        total_possible_pairs += len(unique_pairs)

    total_drawn_ellipses = len(ellipse_data)

    error_percentage = 100 * (total_possible_pairs - total_drawn_ellipses) / total_possible_pairs if total_possible_pairs > 0 else 0

    print(f"Total possible ellipses: {total_possible_pairs}")
    print(f"Total drawn ellipses   : {total_drawn_ellipses}")
    print(f"Error percentage        : {error_percentage:.2f}%")


    df = pd.DataFrame(ellipse_data)
    df.to_csv(out_csv_path, index=False)
    print(f"  Saved ellipse measurements to {out_csv_path}")

    cv2.imwrite(out_img_path, out_img)
    print(f"  Output image saved to {out_img_path}")