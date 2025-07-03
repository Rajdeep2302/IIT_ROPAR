import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import convolve, label, center_of_mass, binary_erosion, binary_dilation
from scipy.signal import find_peaks
from skimage.morphology import skeletonize, remove_small_objects, closing, opening
from skimage.filters import gaussian, threshold_otsu
from collections import deque
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class BoneAnalyzer:
    def __init__(self, min_bone_area: int = 1000, max_distance: int = 150):
        self.min_bone_area = min_bone_area
        self.max_distance = max_distance
        self.processed_image = None
        self.skeleton = None
        self.nodes = []
        self.endpoints = []
        self.connections = {}
        
    def preprocess_xray(self, image: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
        """
        Advanced preprocessing for X-ray bone images with noise reduction and contrast enhancement
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize intensity
        gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if enhance_contrast:
            clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        
        # Gaussian blur to reduce noise
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        
        # Multiple thresholding approaches
        # Method 1: Otsu's thresholding
        _, thresh_otsu = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        # Method 2: Adaptive thresholding (simplified approach)
        # Calculate local threshold based on mean
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv.filter2D(blurred.astype(np.float32), -1, kernel)
        thresh_adaptive = ((blurred > local_mean + 5) * 255).astype(np.uint8)
        
        # Method 3: Custom threshold based on intensity distribution
        hist = cv.calcHist([blurred], [0], None, [256], [0, 256])
        hist_1d = hist.flatten()
        
        # Find peaks in histogram using scipy
        try:
            peaks, _ = find_peaks(hist_1d, height=np.max(hist_1d)*0.1, distance=20)
            
            if len(peaks) >= 2:
                # Use valley between two highest peaks as threshold
                sorted_peaks = sorted(peaks, key=lambda x: hist_1d[x], reverse=True)
                valley_start = min(sorted_peaks[0], sorted_peaks[1])
                valley_end = max(sorted_peaks[0], sorted_peaks[1])
                valley_region = hist_1d[valley_start:valley_end+1]
                if len(valley_region) > 0:
                    custom_thresh = valley_start + np.argmin(valley_region)
                    _, thresh_custom = cv.threshold(blurred, custom_thresh, 255, cv.THRESH_BINARY)
                else:
                    thresh_custom = thresh_otsu
            else:
                thresh_custom = thresh_otsu
        except:
            # Fallback to Otsu if peak detection fails
            thresh_custom = thresh_otsu
        
        # Combine thresholding methods
        combined_thresh = cv.bitwise_or(thresh_otsu, cv.bitwise_and(thresh_adaptive, thresh_custom))
        
        # Morphological operations to clean up
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        cleaned = cv.morphologyEx(combined_thresh, cv.MORPH_CLOSE, kernel)
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_OPEN, kernel)
        
        # Remove small objects
        cleaned_bool = cleaned > 0
        cleaned_bool = remove_small_objects(cleaned_bool, min_size=self.min_bone_area)
        
        self.processed_image = cleaned_bool.astype(np.uint8) * 255
        return self.processed_image
    
    def advanced_skeletonization(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Advanced skeletonization with preprocessing to ensure better connectivity
        """
        # Convert to boolean
        binary_bool = binary_image > 0
        
        # Apply slight dilation to ensure connectivity
        kernel = np.ones((3, 3), np.uint8)
        dilated = binary_dilation(binary_bool, kernel)
        
        # Skeletonize
        skeleton = skeletonize(dilated)
        
        # Clean skeleton - remove small isolated segments
        labeled_skel, num_labels = label(skeleton)
        for i in range(1, num_labels + 1):
            component = labeled_skel == i
            if np.sum(component) < 20:  # Remove very small components
                skeleton[component] = False
        
        self.skeleton = skeleton.astype(np.uint8)
        return self.skeleton
    
    def enhanced_node_detection(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        Enhanced node detection with better handling of junction points
        """
        # Create neighbor count kernel
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        # Count neighbors for each skeleton pixel
        neighbor_count = convolve(skeleton, kernel, mode='constant', cval=0)
        
        # Find junction points (nodes with 3+ neighbors)
        junction_mask = (skeleton == 1) & (neighbor_count >= 3)
        
        # Use more sophisticated clustering for nearby junction points
        labeled_junctions, num_junctions = label(junction_mask, structure=np.ones((3, 3)))
        
        nodes = []
        for i in range(1, num_junctions + 1):
            component = labeled_junctions == i
            # Use center of mass for better node positioning
            centroid = center_of_mass(component)
            nodes.append((int(round(centroid[0])), int(round(centroid[1]))))
        
        self.nodes = nodes
        return nodes
    
    def find_enhanced_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find endpoints with better filtering
        """
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        neighbor_count = convolve(skeleton, kernel, mode='constant', cval=0)
        
        # Endpoints have exactly 1 neighbor
        endpoint_mask = (skeleton == 1) & (neighbor_count == 1)
        endpoints = [tuple(p) for p in np.argwhere(endpoint_mask)]
        
        self.endpoints = endpoints
        return endpoints
    
    def trace_path_with_length(self, skeleton: np.ndarray, start: Tuple[int, int], 
                              end: Tuple[int, int], max_distance: int) -> Tuple[Optional[List[Tuple[int, int]]], float]:
        """
        Trace path between two points and return path with actual length
        """
        h, w = skeleton.shape
        visited = set()
        queue = deque([(start[0], start[1], 0, [start])])
        
        while queue:
            r, c, dist, path = queue.popleft()
            
            if dist > max_distance:
                continue
                
            if (r, c) == end:
                # Calculate actual path length
                path_length = 0
                for i in range(len(path) - 1):
                    path_length += np.sqrt((path[i][0] - path[i+1][0])**2 + 
                                         (path[i][1] - path[i+1][1])**2)
                return path, path_length
            
            if (r, c) in visited:
                continue
                
            visited.add((r, c))
            
            # Check 8-connected neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w and 
                        skeleton[nr, nc] and (nr, nc) not in visited):
                        new_path = path + [(nr, nc)]
                        queue.append((nr, nc, dist + 1, new_path))
        
        return None, 0.0
    
    def find_bone_connections(self, skeleton: np.ndarray, nodes: List[Tuple[int, int]], 
                            endpoints: List[Tuple[int, int]]) -> Dict[int, List[Dict]]:
        """
        Find connections between nodes and endpoints with detailed information
        """
        all_points = nodes + endpoints
        point_types = ['node'] * len(nodes) + ['endpoint'] * len(endpoints)
        connections = {i: [] for i in range(len(all_points))}
        
        # Find connections between all points
        for i in range(len(all_points)):
            for j in range(i + 1, len(all_points)):
                path, length = self.trace_path_with_length(skeleton, all_points[i], 
                                                         all_points[j], self.max_distance)
                if path is not None:
                    connection_info = {
                        'target_index': j,
                        'target_point': all_points[j],
                        'target_type': point_types[j],
                        'path': path,
                        'length': length,
                        'euclidean_distance': np.sqrt((all_points[i][0] - all_points[j][0])**2 + 
                                                    (all_points[i][1] - all_points[j][1])**2)
                    }
                    connections[i].append(connection_info)
                    
                    # Add reverse connection
                    reverse_connection = {
                        'target_index': i,
                        'target_point': all_points[i],
                        'target_type': point_types[i],
                        'path': path[::-1],
                        'length': length,
                        'euclidean_distance': connection_info['euclidean_distance']
                    }
                    connections[j].append(reverse_connection)
        
        self.connections = connections
        return connections
    
    def analyze_bone_structure(self, image: np.ndarray, visualize: bool = True) -> Dict:
        """
        Complete bone structure analysis pipeline
        """
        # Preprocess the image
        processed = self.preprocess_xray(image)
        
        # Skeletonize
        skeleton = self.advanced_skeletonization(processed)
        
        # Find nodes and endpoints
        nodes = self.enhanced_node_detection(skeleton)
        endpoints = self.find_enhanced_endpoints(skeleton)
        
        # Find connections
        connections = self.find_bone_connections(skeleton, nodes, endpoints)
        
        # Calculate statistics
        total_skeleton_length = np.sum(skeleton)
        num_branches = len([conn for conns in connections.values() for conn in conns]) // 2
        
        results = {
            'processed_image': processed,
            'skeleton': skeleton,
            'nodes': nodes,
            'endpoints': endpoints,
            'connections': connections,
            'statistics': {
                'total_skeleton_pixels': total_skeleton_length,
                'num_nodes': len(nodes),
                'num_endpoints': len(endpoints),
                'num_branches': num_branches,
                'avg_branch_length': np.mean([conn['length'] for conns in connections.values() 
                                            for conn in conns]) if num_branches > 0 else 0
            }
        }
        
        if visualize:
            self.visualize_results(image, results)
        
        return results
    
    def visualize_results(self, original_image: np.ndarray, results: Dict):
        """
        Comprehensive visualization of analysis results
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        if len(original_image.shape) == 3:
            axes[0, 0].imshow(cv.cvtColor(original_image, cv.COLOR_BGR2RGB))
        else:
            axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original X-ray Image')
        axes[0, 0].axis('off')
        
        # Processed binary image
        axes[0, 1].imshow(results['processed_image'], cmap='gray')
        axes[0, 1].set_title('Processed Binary Image')
        axes[0, 1].axis('off')
        
        # Skeleton
        axes[0, 2].imshow(results['skeleton'], cmap='gray')
        axes[0, 2].set_title('Skeleton')
        axes[0, 2].axis('off')
        
        # Overlay with nodes and endpoints
        overlay = np.zeros((*results['skeleton'].shape, 3))
        overlay[results['skeleton'] == 1] = [1, 1, 1]  # White skeleton
        
        # Mark nodes in red
        for node in results['nodes']:
            cv.circle(overlay, (node[1], node[0]), 3, (1, 0, 0), -1)
        
        # Mark endpoints in blue
        for endpoint in results['endpoints']:
            cv.circle(overlay, (endpoint[1], endpoint[0]), 3, (0, 0, 1), -1)
        
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Nodes (Red) and Endpoints (Blue)')
        axes[1, 0].axis('off')
        
        # Connection visualization
        connection_overlay = overlay.copy()
        colors = [(0, 1, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)]  # Different colors for connections
        
        connection_count = 0
        for i, connections in results['connections'].items():
            for conn in connections:
                if conn['target_index'] > i:  # Avoid duplicate drawing
                    color = colors[connection_count % len(colors)]
                    path = conn['path']
                    for j in range(len(path) - 1):
                        cv.line(connection_overlay, 
                               (path[j][1], path[j][0]), 
                               (path[j+1][1], path[j+1][0]), 
                               color, 2)
                    connection_count += 1
        
        axes[1, 1].imshow(connection_overlay)
        axes[1, 1].set_title('Bone Connections')
        axes[1, 1].axis('off')
        
        # Statistics
        stats = results['statistics']
        stats_text = f"""
        Bone Structure Analysis Results:
        
        • Total skeleton pixels: {stats['total_skeleton_pixels']}
        • Number of nodes: {stats['num_nodes']}
        • Number of endpoints: {stats['num_endpoints']}
        • Number of branches: {stats['num_branches']}
        • Average branch length: {stats['avg_branch_length']:.2f} pixels
        
        Nodes represent bone junctions/intersections
        Endpoints represent bone terminations
        Branches represent bone segments
        """
        
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Analysis Statistics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def export_measurements(self, results: Dict, filename: str = "bone_measurements.txt"):
        """
        Export detailed measurements to a file
        """
        with open(filename, 'w') as f:
            f.write("BONE STRUCTURE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Statistics
            stats = results['statistics']
            f.write("OVERALL STATISTICS:\n")
            f.write(f"Total skeleton pixels: {stats['total_skeleton_pixels']}\n")
            f.write(f"Number of nodes: {stats['num_nodes']}\n")
            f.write(f"Number of endpoints: {stats['num_endpoints']}\n")
            f.write(f"Number of branches: {stats['num_branches']}\n")
            f.write(f"Average branch length: {stats['avg_branch_length']:.2f} pixels\n\n")
            
            # Detailed connections
            f.write("DETAILED CONNECTIONS:\n")
            all_points = results['nodes'] + results['endpoints']
            point_types = ['node'] * len(results['nodes']) + ['endpoint'] * len(results['endpoints'])
            
            for i, connections in results['connections'].items():
                f.write(f"\nPoint {i+1} ({point_types[i]} at {all_points[i]}):\n")
                for conn in connections:
                    if conn['target_index'] > i:  # Avoid duplicates
                        f.write(f"  -> Point {conn['target_index']+1} ({conn['target_type']})\n")
                        f.write(f"     Path length: {conn['length']:.2f} pixels\n")
                        f.write(f"     Euclidean distance: {conn['euclidean_distance']:.2f} pixels\n")

# Example usage function
def analyze_bone_xray(image_path: str, min_bone_area: int = 1000, max_distance: int = 150):
    """
    Convenient function to analyze bone X-ray from file path
    """
    # Load image
    image = cv.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create analyzer and run analysis
    analyzer = BoneAnalyzer(min_bone_area=min_bone_area, max_distance=max_distance)
    results = analyzer.analyze_bone_structure(image, visualize=True)
    
    # Export measurements
    analyzer.export_measurements(results, f"bone_analysis_{image_path.split('/')[-1]}.txt")
    
    return results, analyzer

# Example of how to use:
# results, analyzer = analyze_bone_xray("path/to/xray.jpg")