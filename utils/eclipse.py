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


def fit_chain_ellipses(region_nodes, region_ends, out_img, ellipse_data, id_start, shape_id, shape_name):
    count = id_start
    visited = set()
    all_points = region_nodes + region_ends

    for p1 in all_points:
        if p1 in visited:
            continue
        candidates = [p2 for p2 in all_points if p2 != p1 and (p1, p2) not in visited and (p2, p1) not in visited]
        if not candidates:
            continue

        p2 = min(candidates, key=lambda x: euclidean_dist(p1, x))

        mid = midpoint(p1, p2)
        major = euclidean_dist(p1, p2) / 2
        minor = max(3, major * 0.5)
        angle = angle_between(p1, p2)

        ellipse_data.append({
            'shape_name': shape_name,
            'shape_id': shape_id,
            'pair_id': count,
            'x': mid[1],
            'y': mid[0],
            'semi_major': major,
            'semi_minor': minor,
            'angle': angle
        })

        cv2.ellipse(out_img, (mid[1], mid[0]), (int(major), int(minor)),
                    angle, 0, 360, (0, 255, 0), 2)

        visited.add(p1)
        visited.add(p2)
        count += 1

    return count


def eclipse_image(img_path, out_img_path, out_csv_path):
    # Prepare input
    bw, img_gray = load_binary(img_path)
    shape_name = os.path.splitext(os.path.basename(img_path))[0]
    labeled, num_shapes = label(bw)
    out_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Detect full skeleton, nodes, and endpoints
    skel = skeletonize(bw).astype(np.uint8)
    nodes = find_nodes(skel)
    endpoints = find_endpoints(skel)

    print(f"  Found {len(nodes)} nodes and {len(endpoints)} endpoints in {shape_name}")
    print(f"  Detected {num_shapes} shapes in image.")

    # Visual markers
    ys, xs = np.where(skel)
    out_img[ys, xs] = (0, 0, 255)

    for r, c in nodes:
        cv2.circle(out_img, (c, r), 3, (0, 255, 0), -1)
    for r, c in endpoints:
        cv2.circle(out_img, (c, r), 3, (255, 0, 0), -1)

    # Begin fitting ellipses
    ellipse_data = []
    global_pair_id = 1
    total_possible_pairs = 0

    for shape_id in range(1, num_shapes + 1):
        mask = labeled == shape_id
        region_nodes = [p for p in nodes if mask[p]]
        region_ends = [p for p in endpoints if mask[p]]

        print(f"    ▶ Shape {shape_id}: {len(region_nodes)} nodes, {len(region_ends)} endpoints")

        global_pair_id = fit_chain_ellipses(
            region_nodes, region_ends, out_img, ellipse_data,
            global_pair_id, shape_id, shape_name
        )

        # Count expected pairs
        unique_pairs = set()
        for i in range(len(region_nodes)):
            for j in range(i + 1, len(region_nodes)):
                unique_pairs.add(tuple(sorted((region_nodes[i], region_nodes[j]))))
        for ept in region_ends:
            if region_nodes:
                closest = min(region_nodes, key=lambda n: euclidean_dist(ept, n))
                unique_pairs.add(tuple(sorted((ept, closest))))
        total_possible_pairs += len(unique_pairs)

        # Annotate shape ID
        ys_s, xs_s = np.where(mask)
        if len(xs_s) > 0 and len(ys_s) > 0:
            cx, cy = int(np.mean(xs_s)), int(np.mean(ys_s))
            cv2.putText(out_img, f'S{shape_id}', (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    total_drawn_ellipses = len(ellipse_data)
    error_percentage = 100 * (total_possible_pairs - total_drawn_ellipses) / total_possible_pairs if total_possible_pairs > 0 else 0

    print(f"✅ Total possible ellipse connections : {total_possible_pairs}")
    print(f"✅ Total ellipses drawn               : {total_drawn_ellipses}")
    print(f"⚠️  Error percentage                   : {error_percentage:.2f}%")

    # Save outputs
    df = pd.DataFrame(ellipse_data)
    df.to_csv(out_csv_path, index=False)
    print(f"📄 Ellipse data saved to {out_csv_path}")

    cv2.imwrite(out_img_path, out_img)
    print(f"🖼️  Output image saved to {out_img_path}")
