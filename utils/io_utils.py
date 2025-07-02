import os
import cv2 as cv
from .preprocess import resize_image, convert_image
from .skeleton import Core_code

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

    Core_code(binary_image_path, "skeletonise_image.csv", "skeletonise_image.png")

def folder_image():
    folder_path = input("Enter path to folder: ").strip()
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return

    os.makedirs("new_resize", exist_ok=True)
    os.makedirs("new_binary", exist_ok=True)
    os.makedirs("new_skeletonise/image", exist_ok=True)
    os.makedirs("new_skeletonise/csv", exist_ok=True)

    for idx, filename in enumerate(os.listdir(folder_path), start=1):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        print(f"\nProcessing image {idx}: {filename}")
        image_path = os.path.join(folder_path, filename)
        image = cv.imread(image_path)
        if image is None:
            print(f"Skipping invalid image: {filename}")
            continue
        resized_path = f"new_resize/resize_image_{idx}.png"
        bin_path = f"new_binary/binary_image_{idx}.png"
        csv_path = f"new_skeletonise/csv/skeletonise_image_{idx}.csv"
        out_path = f"new_skeletonise/image/skeletonise_image_{idx}.png"

        image = resize_image(image, resized_path)
        binary_image = convert_image(image)
        cv.imwrite(bin_path, binary_image)
        Core_code(bin_path, csv_path, out_path)

    print("\nAll images processed.")
