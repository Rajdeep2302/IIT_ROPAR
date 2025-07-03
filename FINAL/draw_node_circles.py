import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.feature import corner_peaks
from skimage.graph import route_through_array
from scipy.ndimage import distance_transform_edt
import pandas as pd

# Input and output directories
INPUT_DIR = 'binary/'
OUTPUT_IMG_DIR = 'output/circles_all/'
OUTPUT_CSV_DIR = 'output/circles_all_csv/'
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

def detect_nodes(skel_img):
    # Use corner_peaks to find node-like points on the skeleton
    # You can adjust min_distance as needed
    coords = corner_peaks(skel_img.astype(np.uint8), min_distance=5, threshold_rel=0.1)
    return coords

def geodesic_radius(binary_img, node):
    # Invert image: white=1, black=0
    mask = (binary_img > 0).astype(np.uint8)
    # Compute distance transform from black pixels (boundary)
    dist = distance_transform_edt(mask)
    # The value at the node is the Euclidean distance to the nearest black pixel
    # For geodesic, we need to follow the white region only
    # We'll use a BFS or Dijkstra approach, but for now, use Euclidean as a fallback
    y, x = node
    radius = dist[y, x]
    return radius

def process_image(img_path, out_img_path, out_csv_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load {img_path}")
        return
    # Binarize if not already
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Skeletonize
    skel = skeletonize(binary // 255)
    # Detect nodes
    nodes = detect_nodes(skel)
    # Compute geodesic radius for each node
    radii = []
    out_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for idx, (y, x) in enumerate(nodes):
        radius = geodesic_radius(binary, (y, x))
        radii.append({'node_id': idx, 'x': x, 'y': y, 'radius': radius})
        # Draw circle
        cv2.circle(out_img, (x, y), int(radius), (0, 0, 255), 1)
        cv2.circle(out_img, (x, y), 2, (0, 255, 0), -1)  # mark node center
    # Save output image
    cv2.imwrite(out_img_path, out_img)
    # Save radii to CSV
    df = pd.DataFrame(radii)
    df.to_csv(out_csv_path, index=False)

def main():
    for fname in os.listdir(INPUT_DIR):
        if fname.endswith('.png'):
            img_path = os.path.join(INPUT_DIR, fname)
            out_img_path = os.path.join(OUTPUT_IMG_DIR, fname)
            out_csv_path = os.path.join(OUTPUT_CSV_DIR, fname.replace('.png', '.csv'))
            print(f"Processing {fname}...")
            process_image(img_path, out_img_path, out_csv_path)

def draw_circles_on_nodes(binary_image_path, output_image_path, nodes):
    # Load binary image (white=1, black=0)
    img = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    _, bw = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    dist_transform = distance_transform_edt(bw)
    # Load the output image to draw on
    out_img = cv2.imread(output_image_path)
    for (r, c) in nodes:
        radius = int(dist_transform[r, c])
        if radius > 0:
            cv2.circle(out_img, (c, r), radius, (0, 0, 255), 1)  # Red circle
    # Save new image
    cv2.imwrite(output_image_path.replace('.png', '_with_circles.png'), out_img)

if __name__ == '__main__':
    main() 