import cv2 as cv
import numpy as np

def resize_image(image: np.ndarray, resize_image_path: str, scale: int = 8) -> np.ndarray:
    img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    h, w, _ = img.shape
    resized = cv.resize(img, (w * scale, h * scale), interpolation=cv.INTER_LANCZOS4)
    cv.imwrite(resize_image_path, cv.cvtColor(resized, cv.COLOR_RGB2BGR))
    return resized

def convert_image(image: np.ndarray) -> np.ndarray:
    # Apply Gaussian blur to smooth edges
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 1.0)
    
    # Adaptive thresholding for better bone segmentation
    thresh = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations with smaller kernel and fewer iterations to minimize erosion
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
    thresh = cv.dilate(thresh, kernel, iterations=1)  # Add dilation to recover edges
    
    return thresh