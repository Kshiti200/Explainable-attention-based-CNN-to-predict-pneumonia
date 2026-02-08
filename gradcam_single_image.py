import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from preprocessing import apply_clahe

IMG_SIZE = 224

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model("attention_pneumonia_model.h5")
print("✅ Model Loaded Successfully!")


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = apply_clahe(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# -----------------------------
# Grad-CAM function
# -----------------------------
def generate_gradcam(img_path, last_conv_layer_name="conv2d_3"):
    img = preprocess_image(img_path)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap.numpy()


# -----------------------------
# Overlay heatmap on image
# -----------------------------
def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlayed_img


# -----------------------------
# MAIN
# -----------------------------
image_path = r"C:\Users\Kshiti\Desktop\Pneumonia\chest_xray\test\PNEUMONIA\person1_virus_6.jpeg"

heatmap = generate_gradcam(image_path)
gradcam_img = overlay_gradcam(image_path, heatmap)

# Show output
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM Visualization (Pneumonia)")
plt.axis("off")
plt.show()

# Save output
cv2.imwrite("gradcam_output.png", gradcam_img)
print("✅ Grad-CAM image saved as gradcam_output.png")
