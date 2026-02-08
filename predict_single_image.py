import tensorflow as tf
import numpy as np
import cv2
import os

from preprocessing import apply_clahe

IMG_SIZE = 224

#  Load trained model
model = tf.keras.models.load_model("attention_pneumonia_model.h5")
print("Model Loaded Successfully!")


#  Step 1: Preprocess the input image
def preprocess_image(img_path):
    img = cv2.imread(img_path)

    # Resize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Apply CLAHE enhancement
    img = apply_clahe(img)

    # Normalize pixel values
    img = img / 255.0

    # Expand dimensions for model input
    img = np.expand_dims(img, axis=0)

    return img


# Step 2: Predict image class
def predict_image(img_path):
    img = preprocess_image(img_path)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        predicted_label = "PNEUMONIA"
    else:
        predicted_label = "NORMAL"

    return predicted_label, prediction[0][0]


#  Step 3: Check prediction correctness
def check_prediction(img_path):
    predicted_label, confidence = predict_image(img_path)

    # Ground truth label from folder name
    actual_label = os.path.basename(os.path.dirname(img_path)).upper()

    print("\n Image Path:", img_path)
    print("Actual Label:", actual_label)
    print("Predicted Label:", predicted_label)
    print("Confidence Score:", round(confidence, 4))

    if predicted_label == actual_label:
        print("Prediction is CORRECT")
    else:
        print("Prediction is INCORRECT")


#  Step 4: Example Usage (Your Image Path Updated)
image_path = r"C:\Users\Kshiti\Desktop\Pneumonia\chest_xray\test\PNEUMONIA\person1_virus_7.jpeg"

check_prediction(image_path)
