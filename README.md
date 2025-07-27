# 🦴 TABULATOR: Bone Topology Analysis and Segmentation

## 🚀 Introduction

**TABULATOR** is a robust tool for detailed bone topology analysis using advanced image segmentation techniques. Designed for researchers, clinicians, and data scientists, it processes bone scan images to identify and quantify key structural features—delivering precise, quantitative insights into bone architecture from medical imaging data.

---

## ✨ Features

- **🖼️ Image Preprocessing:** Resizes and converts images to high-contrast binary format for optimal analysis.
- **💀 Skeletonization:** Uses scikit-image’s `skeletonize` to reduce bone structures to their essential topological skeleton.
- **📍 Node & Endpoint Detection:** Detects critical points—junctions (`find_nodes`) and terminations (`find_endpoints`)—within the skeleton.
- **🔗 Topological Analysis:** Calculates Euclidean distances between nodes and matches endpoints to their nearest nodes.
- **📐 Geometric Measurements:** Computes bone fragment volumes and fits ellipses to estimate thickness and geometry.
- **📊 Data Export:** Saves all extracted data as CSV files for further statistical analysis.
- **🎨 Advanced Visualization:** Generates annotated images showing skeletons, nodes, endpoints, and fitted shapes.

---

## ⚙️ Workflow Overview

1. **Image Loading & Preprocessing:**  
   Load a TIFF image, resize to standard dimensions, and convert to binary using thresholding.

2. **Skeletonization:**  
   Create a one-pixel-wide skeleton representation for accurate node and endpoint detection.

3. **Feature Detection:**  
   - **Nodes:** Junctions (pixels with ≥3 neighbors) via convolution.
   - **Endpoints:** Terminations (pixels with 1 neighbor).

4. **Topological & Geometric Analysis:**  
   - Calculate Euclidean distances between nodes.
   - Compute geodesic radius at each node (local bone thickness).
   - Fit ellipses to endpoint pairs to model bone segment shape and orientation.

5. **Data Aggregation & Export:**  
   Compile node coordinates, endpoint locations, distances, and ellipse parameters into CSV files.

---

## 💻 Key Functions

```python
# Node Detection
def find_nodes(skel: np.ndarray) -> List[Tuple[int, int]]:
    deg = neighbour_count(skel)
    node_mask = (skel == 1) & (deg >= 3)
    lbl, n_comp = label(node_mask)
    centroids = center_of_mass(node_mask, lbl, range(1, n_comp + 1))
    return [(int(round(r)), int(round(c))) for r, c in centroids]

# Endpoint Detection
def find_endpoints(skel: np.ndarray) -> List[Tuple[int, int]]:
    deg = neighbour_count(skel)
    return [tuple(p) for p in np.argwhere((skel == 1) & (deg == 1))]

# Geodesic Radius Calculation
def geodesic_radius(binary_img, node):
    mask = (binary_img > 0).astype(np.uint8)
    dist = distance_transform_edt(mask)
    y, x = node
    return dist[y, x]
```

---

## 📦 Dependencies

- numpy
- opencv-python
- scipy
- pandas
- scikit-image
- seaborn
- matplotlib

---

## 🛠️ Installation

```bash
git clone <your-repository-url>
cd <your-repository-name>
pip install numpy opencv-python scipy pandas scikit-image seaborn matplotlib
```

---

## 🚀 Usage

1. **Set Input Image Path:**  
   Update the `input_image` variable in the notebook with your TIFF image path.

2. **Run the Notebook:**  
   Execute all cells in order.

3. **(Optional) Save Intermediate Images:**  
   Choose to save resized and binary images when prompted.

4. **Review Outputs:**  
   All results (annotated images, CSV data) are saved in the `Single_image/` directory.

---

## 📂 Output Files

- `resize_image.png` — Resized input image
- `binary_image.png` — Binary version of the image
- `skeleton_image.png` — Skeletonized bone structures
- `node_image.png` — Annotated nodes and endpoints
- `node_image.csv` — Data on detected nodes, endpoints, and connections
- `circle_image.png` — Circles fitted to detected nodes
- `circle_image.csv` — Data on fitted circles
- `ellipse_image.png` — Ellipses fitted to bone structures
- `ellipse_image.csv` — Data on fitted ellipses
- `volume.csv` — Volume of each detected bone shape
- `summary.csv` — Summary of analysis results

---

## 🤝 Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss improvements.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for