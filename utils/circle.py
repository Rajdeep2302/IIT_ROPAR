import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, convolve, label, center_of_mass
import pandas as pd
from .graph_analysis import *
from .skeleton import *

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

def circle_image(img_path, out_img_path, out_csv_path):
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
    if img_gray is None:
        print(f"❌ Failed to load image for drawing: {img_path}")
        return
    
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