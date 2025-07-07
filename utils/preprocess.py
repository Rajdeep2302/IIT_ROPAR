import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def resize_image(image, resize_image_path):
    img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    h, w, _ = img.shape
    scale = 8
    methods = [cv.INTER_LANCZOS4]
    titles = ["INTER_LANCZOS4"] 

    for i in range(len(methods)):
        resized = cv.resize(img, (w * scale, h * scale), interpolation=methods[i])
        if i == 0:
            best_resize = resized
    cv.imwrite(resize_image_path, cv.cvtColor(best_resize, cv.COLOR_RGB2BGR))
    return best_resize

def convert_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mat = np.array(gray) / 255.0
    mat = np.where(mat < 0.5, 0, 1)
    return (mat * 255).astype(np.uint8)
