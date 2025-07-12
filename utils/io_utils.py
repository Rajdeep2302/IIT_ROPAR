import os
import cv2 as cv
from .preprocess import resize_image, convert_image
from utils.graph_analysis import *
from .skeleton import Core_code
from .circle import *
from .eclipse import *
from .volume import *
from .draw_node_circles import *

def single_file():
    image_path = input("Enter path to image: ").strip()
    if not os.path.exists(image_path):
        print("File does not exist.")
        return
    image = cv.imread(image_path)
    if image is None:
        print("Unable to read the image.")
        return

    resize_image_path = f"resize_image.png"
    image = resize_image(image, resize_image_path)
    binary_image = convert_image(image)
    binary_image_path = "binary_image.png"
    cv.imwrite(binary_image_path, binary_image)

    compute_shape_volumes(binary_image_path, "Volume_file.csv")

    img_path = Core_code(binary_image_path, "skeletonise_image.csv", "skeletonise_image.png")
    circle_image(binary_image_path, "circle_image.png", "circle_image.csv")
    eclipse_image(binary_image_path, "eclipse_image.png", "eclipse_image.csv")
    all_circle(binary_image_path, "all_circle.png", "all_circle.csv","shape/node_circle.csv")

def folder_image():
    folder_path = input("Enter path to folder: ").strip()
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return
    
    os.makedirs("output/new_resize", exist_ok=True)
    os.makedirs("output/new_binary", exist_ok=True)
    os.makedirs("output/new_skeletonise/image", exist_ok=True)
    os.makedirs("output/new_skeletonise/csv", exist_ok=True)
    os.makedirs("output/circles/image", exist_ok=True)
    os.makedirs("output/circles/csv", exist_ok=True)
    os.makedirs("output/eclipse/image", exist_ok=True)
    os.makedirs("output/eclipse/csv", exist_ok=True)
    os.makedirs("output/all_circle/image", exist_ok=True)
    os.makedirs("output/all_circle/csv", exist_ok=True)
    os.makedirs("output/all_circle/shape", exist_ok=True)

    for idx, filename in enumerate(os.listdir(folder_path), start=1):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        print(f"\nProcessing image {idx}: {filename}")
        image_path = os.path.join(folder_path, filename)
        image = cv.imread(image_path)
        if image is None:
            print(f"Skipping invalid image: {filename}")
            continue
        resized_path = f"output/new_resize/resize_image_{idx}.png"
        bin_path = f"output/new_binary/binary_image_{idx}.png"
        scal_csv_path = f"output/new_skeletonise/csv/skeletonise_image_{idx}.csv"
        scal_out_path = f"output/new_skeletonise/image/skeletonise_image_{idx}.png"
        circles_csv_path = f"output/circles/csv/circle_image_{idx}.csv"
        circles_out_path = f"output/circles/image/circle_image_{idx}.png"
        eclipse_csv_path = f"output/eclipse/csv/circle_image_{idx}.csv"
        eclipse_out_path = f"output/eclipse/image/circle_image_{idx}.png"
        all_circle_csv_path = f"output/all_circle/csv/circle_image_{idx}.csv"
        all_circle_shape_csv_path = f"output/all_circle/shape/circle_image_{idx}.csv"
        all_circle_out_path = f"output/all_circle/image/circle_image_{idx}.png"
        volume = f"output/volume_{idx}.png"


        image = resize_image(image, resized_path)
        binary_image = convert_image(image)
        cv.imwrite(bin_path, binary_image)
        compute_shape_volumes(bin_path, volume)
        img_path =  Core_code(bin_path, scal_csv_path, scal_out_path)
        circle_image(bin_path, circles_out_path, circles_csv_path)
        eclipse_image(bin_path, eclipse_out_path, eclipse_csv_path)
        all_circle(bin_path, all_circle_out_path, all_circle_csv_path, all_circle_shape_csv_path)

    print("\nAll images processed.")
