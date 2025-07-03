import os
import cv2 as cv
from .preprocess import resize_image, convert_image
from .skeleton import Core_code

def single_file():
    image_path = input("Enter path to X-ray image: ").strip()
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at '{image_path}'.")
        return

    resize_image_path = "resize_image.png"
    image = resize_image(image, resize_image_path)
    binary_image = convert_image(image)
    binary_image_path = "binary_image.png"
    cv.imwrite(binary_image_path, binary_image)

    Core_code(binary_image_path, "skeletonise_image.csv", "skeletonise_image.png")
    print(f"Processed image saved as 'skeletonise_image.png' and data as 'skeletonise_image.csv'.")

def folder_image():
    folder_path = input("Enter path to folder containing X-ray images: ").strip()
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Error: Invalid folder path '{folder_path}'.")
        return

    os.makedirs("output/resize", exist_ok=True)
    os.makedirs("output/binary", exist_ok=True)
    os.makedirs("output/skeleton/image", exist_ok=True)
    os.makedirs("output/skeleton/csv", exist_ok=True)

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    processed_count = 0
    for idx, filename in enumerate(os.listdir(folder_path), start=1):
        if not filename.lower().endswith(image_extensions):
            continue
        print(f"\nProcessing image {idx}: {filename}")
        image_path = os.path.join(folder_path, filename)
        image = cv.imread(image_path)
        if image is None:
            print(f"Warning: Skipping invalid image '{filename}'.")
            continue
        resized_path = f"output/resize/resize_image_{idx}.png"
        bin_path = f"output/binary/binary_image_{idx}.png"
        csv_path = f"output/skeleton/csv/skeletonise_image_{idx}.csv"
        out_path = f"output/skeleton/image/skeletonise_image_{idx}.png"

        image = resize_image(image, resized_path)
        binary_image = convert_image(image)
        cv.imwrite(bin_path, binary_image)
        Core_code(bin_path, csv_path, out_path)
        processed_count += 1

    print(f"\nProcessed {processed_count} images. Outputs saved in 'output' directory.")