import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.feature import corner_peaks
from scipy.ndimage import distance_transform_edt, label
import pandas as pd
import random
from skimage.graph import route_through_array

def connect_nodes_with_geodesic_lines(image, skeleton, nodes, color=(0, 255, 255)):
    skeleton = skeleton.astype(np.uint8)
    cost_map = 1.0 - skeleton  # low cost where skeleton is present

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            start = tuple(nodes[i])
            end = tuple(nodes[j])

            # Use geodesic route through the skeleton
            try:
                path, cost = route_through_array(cost_map, start, end, fully_connected=True)
                path = np.array(path)
                if len(path) < 2 or cost > 1.5 * np.linalg.norm(np.array(start) - np.array(end)):
                    continue  # Skip very long or invalid connections

                # Draw the path in yellow
                for k in range(len(path) - 1):
                    pt1 = (path[k][1], path[k][0])
                    pt2 = (path[k + 1][1], path[k + 1][0])
                    cv2.line(image, pt1, pt2, color, 1)
            except:
                continue


def detect_nodes(skel_img):
    coords = corner_peaks(skel_img.astype(np.uint8), min_distance=5, threshold_rel=0.1)
    return coords

def geodesic_radius(binary_img, node):
    mask = (binary_img > 0).astype(np.uint8)
    dist = distance_transform_edt(mask)
    y, x = node
    return dist[y, x]

def generate_unique_color(existing_colors):
    while True:
        color = tuple(random.randint(50, 255) for _ in range(3))
        if color not in existing_colors and color not in [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]:  # Avoid red, green, blue, yellow
            return color

def all_circle(img_path, out_img_path, out_csv_path, shape_csv):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load {img_path}")
        return

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    labeled, num_shapes = label(binary)

    final_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    shape_data = []
    node_data = []
    existing_colors = set()
    node_id_counter = 0

    base_path = os.path.splitext(shape_csv)[0]

    for shape_id in range(1, num_shapes + 1):
        mask = (labeled == shape_id).astype(np.uint8) * 255
        skel = skeletonize(mask // 255)

        nodes = detect_nodes(skel)
        color = generate_unique_color(existing_colors)
        existing_colors.add(color)

        shape_length = np.count_nonzero(skel)

        sorted_nodes = sorted(nodes, key=lambda p: (p[0], p[1]))  # y, x order

        per_shape_nodes = []

        for idx, (y, x) in enumerate(sorted_nodes):
            radius = geodesic_radius(mask, (y, x))
            node_info = {
                'node_id': node_id_counter,
                'shape_id': shape_id,
                'x': x,
                'y': y,
                'radius': radius
            }
            node_data.append(node_info)
            per_shape_nodes.append(node_info)

            # Draw radius circle
            cv2.circle(final_img, (x, y), int(radius), color, 1)
            cv2.circle(final_img, (x, y), 2, (0, 255, 0), -1)
            node_id_counter += 1

        # Save per-shape node data
        per_shape_df = pd.DataFrame(per_shape_nodes)
        shape_csv_path = f"{base_path}_shape_{shape_id}_nodes.csv"
        per_shape_df.to_csv(shape_csv_path, index=False)

        connect_nodes_with_geodesic_lines(final_img, skel, sorted_nodes, color=(255, 0, 0))

        ys, xs = np.nonzero(mask)
        if len(xs) > 0 and len(ys) > 0:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            cv2.putText(final_img, f'S{shape_id}', (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 255), 2, cv2.LINE_AA)

        shape_data.append({
            'shape_id': shape_id,
            'num_nodes': len(nodes),
            'skeleton_length': shape_length
        })

    # Save overall outputs
    cv2.imwrite(out_img_path, final_img)
    pd.DataFrame(node_data).to_csv(out_csv_path.replace('.csv', '_nodes.csv'), index=False)
    pd.DataFrame(shape_data).to_csv(out_csv_path.replace('.csv', '_shapes.csv'), index=False)
