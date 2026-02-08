import cv2
import numpy as np

def apply_clahe(img):
    """
    CLAHE that is compatible with Keras ImageDataGenerator
    """

    # Ensure uint8 (0–255)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)

    # Convert back to RGB (for CNN compatibility)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return enhanced
