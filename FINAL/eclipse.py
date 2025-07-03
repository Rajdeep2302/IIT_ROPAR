import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, convolve, label, center_of_mass
import pandas as pd

# Input and output directories
INPUT_DIR = 'binary/'
OUTPUT_IMG_DIR = 'output/eclipse/'
OUTPUT_CSV_DIR = 'output/eclipse_csv/'
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

def calculate_ellipse_parameters(binary_img, node):
    """Calculate ellipse parameters that touch at least 2 closest borders"""
    y, x = node
    
    # Get image dimensions
    h, w = binary_img.shape
    
    # First, use distance transform as a fallback
    mask = (binary_img > 0).astype(np.uint8)
    dist_transform = distance_transform_edt(mask)
    fallback_radius = dist_transform[y, x]
    
    # Sample distances in 16 directions (every 22.5 degrees) for better accuracy
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    distances = []
    
    for angle in angles:
        dy = np.sin(angle)
        dx = np.cos(angle)
        
        distance = 0
        cy, cx = float(y), float(x)
        
        # Walk in this direction until we hit a black pixel or boundary
        while True:
            cy += dy
            cx += dx
            distance += 1
            
            # Check bounds
            iy, ix = int(round(cy)), int(round(cx))
            if (iy < 0 or iy >= h or ix < 0 or ix >= w or 
                binary_img[iy, ix] == 0):
                break
                
            # Safety check to prevent infinite loops
            if distance > max(h, w):
                break
        
        distances.append(max(1, distance))  # Ensure minimum distance of 1
    
    # Convert to numpy array for easier manipulation
    distances = np.array(distances)
    
    # If all distances are very small, use fallback
    if np.max(distances) < 3:
        return max(3, fallback_radius), max(2, fallback_radius * 0.7), 0
    
    # Find the two smallest distances for the ellipse axes
    sorted_distances = np.sort(distances)
    
    # Use the two smallest distances, but ensure reasonable minimum sizes
    semi_minor = max(2, sorted_distances[0])
    semi_major = max(semi_minor + 1, sorted_distances[1])
    
    # Find the angle of the major axis
    # Get the direction corresponding to the minimum distance
    min_idx = np.argmin(distances)
    major_angle = angles[min_idx] * 180 / np.pi  # Convert to degrees
    
    # The major axis should be perpendicular to the minimum distance direction
    angle = (major_angle + 90) % 180
    
    return semi_major, semi_minor, angle

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
    
    # Debug: Print node coordinates
    for idx, (r, c) in enumerate(nodes):
        print(f"    Node {idx+1}: ({r}, {c})")
    
    # Load original image for visualization
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert to color for output
    out_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # Draw skeleton in red
    ys, xs = np.where(skel)
    out_img[ys, xs] = (0, 0, 255)  # Red for skeleton
    
    # Compute ellipse parameters for each node
    ellipse_data = []
    
    # Process nodes - use same indexing as skeleton.py (1-based)
    for idx, (r, c) in enumerate(nodes):
        try:
            # Convert back to 255-scale binary for ellipse calculation
            binary_255 = (bw * 255).astype(np.uint8)
            semi_major, semi_minor, angle = calculate_ellipse_parameters(binary_255, (r, c))
            
            ellipse_data.append({
                'node_id': idx + 1, 
                'x': c, 
                'y': r, 
                'semi_major': semi_major,
                'semi_minor': semi_minor,
                'angle': angle
            })
            
            # Draw ellipse around node
            cv2.ellipse(out_img, (c, r), (int(semi_major), int(semi_minor)), 
                       angle, 0, 360, (0, 255, 0), 2)  # Green ellipse
            cv2.circle(out_img, (c, r), 5, (0, 255, 0), -1)  # Green dot for node center
            
            # Add node label (same as skeleton.py)
            cv2.putText(out_img, str(idx + 1), (c + 8, r - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)  # Yellow text
            
            print(f"    Node {idx+1}: ellipse ({semi_major:.1f}, {semi_minor:.1f}) angle={angle:.1f}°")
            
        except Exception as e:
            print(f"    Error processing node {idx+1} at ({r}, {c}): {e}")
            # Fallback: draw a small circle
            cv2.circle(out_img, (c, r), 10, (0, 255, 0), 2)
            cv2.circle(out_img, (c, r), 5, (0, 255, 0), -1)
    
    # Draw endpoints (same as skeleton.py)
    for idx, (r, c) in enumerate(endpoints):
        cv2.circle(out_img, (c, r), 5, (255, 0, 0), -1)  # Blue for endpoints
        label = chr(ord('A') + idx)
        cv2.putText(out_img, label, (c + 8, r - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)  # Cyan text
    
    # Save output image
    cv2.imwrite(out_img_path, out_img)
    
    # Save ellipse data to CSV
    if ellipse_data:
        df = pd.DataFrame(ellipse_data)
        df.to_csv(out_csv_path, index=False)
        print(f"  Saved {len(ellipse_data)} node ellipse measurements to {out_csv_path}")
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