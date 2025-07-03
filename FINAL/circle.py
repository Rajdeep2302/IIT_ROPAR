import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, convolve, label, center_of_mass
import pandas as pd

# Input and output directories
INPUT_DIR = 'binary/'
OUTPUT_IMG_DIR = 'output/circles/'
OUTPUT_CSV_DIR = 'output/circles_csv/'
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

def skeletonise_image(bw: np.ndarray) -> np.ndarray:
    """Same as graph_analysis.py"""
    return skeletonize(bw).astype(np.uint8)

def neighbour_count(skel: np.ndarray) -> np.ndarray:
    """Same as graph_analysis.py"""
    kernel = np.ones((3, 3), np.uint8)
    return convolve(skel, kernel, mode="constant", cval=0) - skel

def find_endpoints(skel: np.ndarray) -> list:
    """Same as graph_analysis.py"""
    deg = neighbour_count(skel)
    return [tuple(p) for p in np.argwhere((skel == 1) & (deg == 1))]

def find_nodes(skel: np.ndarray) -> list:
    """Exact same function as graph_analysis.py"""
    deg = neighbour_count(skel)
    node_mask = (skel == 1) & (deg >= 3)
    lbl, n_comp = label(node_mask)
    centroids = center_of_mass(node_mask, lbl, range(1, n_comp + 1))
    return [(int(round(r)), int(round(c))) for r, c in centroids]

def geodesic_radius(binary_img, node):
    """Calculate the distance from node to the nearest background pixel"""
    # Invert image: white=1, black=0
    mask = (binary_img > 0).astype(np.uint8)
    # Compute distance transform from black pixels (boundary)
    dist = distance_transform_edt(mask)
    # The value at the node is the Euclidean distance to the nearest black pixel
    y, x = node
    radius = dist[y, x]
    return radius

def load_binary(path: str) -> np.ndarray:
    """Same as skeleton.py"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, bw = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return bw.astype(np.uint8)

def process_image(img_path, out_img_path, out_csv_path):
    # Load binary image using same method as skeleton.py
    bw = load_binary(img_path)
    
    # Skeletonize using same method as graph_analysis.py
    skel = skeletonise_image(bw)
    
    # Find nodes using EXACT same method as graph_analysis.py
    nodes = find_nodes(skel)
    endpoints = find_endpoints(skel)
    
    print(f"  Found {len(nodes)} nodes and {len(endpoints)} endpoints")
    
    # Load original image for visualization
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert to color for output
    out_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # Draw skeleton in red
    ys, xs = np.where(skel)
    out_img[ys, xs] = (0, 0, 255)  # Red for skeleton
    
    # Compute geodesic radius for each node
    radii = []
    
    # Process nodes - use same indexing as skeleton.py (1-based)
    for idx, (r, c) in enumerate(nodes):
        # Convert back to 255-scale binary for radius calculation
        binary_255 = (bw * 255).astype(np.uint8)
        radius = geodesic_radius(binary_255, (r, c))
        radii.append({'node_id': idx + 1, 'x': c, 'y': r, 'radius': radius})
        
        # Draw circle around node
        cv2.circle(out_img, (c, r), int(radius), (0, 255, 0), 2)  # Green circle
        cv2.circle(out_img, (c, r), 5, (0, 255, 0), -1)  # Green dot for node center
        
        # Add node label (same as skeleton.py)
        cv2.putText(out_img, str(idx + 1), (c + 8, r - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)  # Yellow text
    
    # Draw endpoints (same as skeleton.py)
    for idx, (r, c) in enumerate(endpoints):
        cv2.circle(out_img, (c, r), 5, (255, 0, 0), -1)  # Blue for endpoints
        label = chr(ord('A') + idx)
        cv2.putText(out_img, label, (c + 8, r - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)  # Cyan text
    
    # Save output image
    cv2.imwrite(out_img_path, out_img)
    
    # Save radii to CSV
    if radii:
        df = pd.DataFrame(radii)
        df.to_csv(out_csv_path, index=False)
        print(f"  Saved {len(radii)} node measurements to {out_csv_path}")
    else:
        print(f"  No nodes found in {img_path}")

def main():
    for fname in os.listdir(INPUT_DIR):
        if fname.endswith('.png'):
            img_path = os.path.join(INPUT_DIR, fname)
            out_img_path = os.path.join(OUTPUT_IMG_DIR, fname)
            out_csv_path = os.path.join(OUTPUT_CSV_DIR, fname.replace('.png', '.csv'))
            print(f"Processing {fname}...")
            process_image(img_path, out_img_path, out_csv_path)

if __name__ == '__main__':
    main()