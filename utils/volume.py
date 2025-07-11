import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import label

def compute_shape_volumes(img_path, out_csv_path, pixel_area=1.0):
    """
    Computes the volume (area) of each white shape in a binary image.
    
    Args:
        img_path: path to the binary image
        out_csv_path: path to save the output CSV
        pixel_area: real-world area per pixel (default = 1.0)
    """
    # Load grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image from {img_path}")

    # Convert to binary (white=1, black=0)
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

    # Label connected components
    labeled, num_shapes = label(binary)

    shape_areas = []

    for shape_id in range(1, num_shapes + 1):
        shape_mask = labeled == shape_id
        pixel_count = np.count_nonzero(shape_mask)
        volume = pixel_count * pixel_area

        shape_areas.append({
            'shape_id': shape_id,
            'pixel_count': pixel_count,
            'volume': volume
        })

    # Save to CSV
    df = pd.DataFrame(shape_areas)
    df.to_csv(out_csv_path, index=False)

    print(f"✅ Found {num_shapes} shapes")
    print(f"📄 Volume data saved to {out_csv_path}")
