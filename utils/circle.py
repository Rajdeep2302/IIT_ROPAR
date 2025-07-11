import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, label
import pandas as pd
from .graph_analysis import *
from .skeleton import *

def geodesic_radius(binary_img, node):
    mask = (binary_img > 0).astype(np.uint8)
    dist = distance_transform_edt(mask)
    y, x = node
    return dist[y, x]

def circle_image(img_path, out_img_path, out_csv_path):
    # Load binary image
    bw = load_binary(img_path)
    shape_name = os.path.splitext(os.path.basename(img_path))[0]

    # Label connected components
    labeled, num_shapes = label(bw)
    out_img = cv2.cvtColor((bw * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    node_data = []

    node_id_counter = 1  # Global node ID

    print(f"🔎 Found {num_shapes} shapes in {shape_name}")

    for shape_id in range(1, num_shapes + 1):
        mask = (labeled == shape_id).astype(np.uint8)

        # Skeletonize shape
        skel = skeletonise_image(mask)

        # Find nodes and endpoints
        nodes = find_nodes(skel)
        endpoints = find_endpoints(skel)

        print(f"  ▶ Shape {shape_id}: {len(nodes)} nodes, {len(endpoints)} endpoints")

        # Draw skeleton in red
        ys, xs = np.where(skel)
        out_img[ys, xs] = (0, 0, 255)  # Red

        for idx, (r, c) in enumerate(nodes):
            binary_255 = (mask * 255).astype(np.uint8)
            radius = geodesic_radius(binary_255, (r, c))

            node_data.append({
                'shape_name': shape_name,
                'shape_id': shape_id,
                'node_id': node_id_counter,
                'x': c,
                'y': r,
                'radius': radius
            })

            # Draw node and radius
            cv2.circle(out_img, (c, r), int(radius), (0, 255, 0), 2)
            cv2.circle(out_img, (c, r), 5, (0, 255, 0), -1)
            cv2.putText(out_img, str(node_id_counter), (c + 8, r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            node_id_counter += 1

        # Draw endpoints
        for ep_idx, (r, c) in enumerate(endpoints):
            label_txt = chr(ord('A') + ep_idx)
            cv2.circle(out_img, (c, r), 5, (255, 0, 0), -1)
            cv2.putText(out_img, label_txt, (c + 8, r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Label shape ID in center
        ys_s, xs_s = np.where(mask)
        if len(xs_s) > 0 and len(ys_s) > 0:
            cx, cy = int(np.mean(xs_s)), int(np.mean(ys_s))
            cv2.putText(out_img, f'S{shape_id}', (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Save final annotated image
    cv2.imwrite(out_img_path, out_img)

    # Save node data to CSV
    if node_data:
        df = pd.DataFrame(node_data)
        df.to_csv(out_csv_path, index=False)
        print(f"✅ Saved {len(node_data)} node entries to {out_csv_path}")
    else:
        print("⚠️ No nodes detected.")
