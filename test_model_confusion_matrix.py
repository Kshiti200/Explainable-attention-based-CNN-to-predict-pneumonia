import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------
# Step 1: Paths
# ----------------------------
DATA_DIR = "chest_xray"
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SIZE = 224
BATCH_SIZE = 16

# ----------------------------
# Step 2: Load trained model
# ----------------------------
model = tf.keras.models.load_model("attention_pneumonia_model.h5")
print(" Trained model loaded successfully!")

# ----------------------------
# Step 3: Load test dataset
# ----------------------------
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ----------------------------
# Step 4: Evaluate model
# ----------------------------
test_loss, test_accuracy = model.evaluate(test_data)

print(f"\n Test Accuracy: {test_accuracy * 100:.2f}%")
print(f" Test Loss: {test_loss:.4f}")

# ----------------------------
# Step 5: Predictions
# ----------------------------
y_pred_probs = model.predict(test_data)
y_pred = (y_pred_probs > 0.5).astype(int)

y_true = test_data.classes

# ----------------------------
# Step 6: Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

# ----------------------------
# Step 7: Classification Report
# ----------------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

# ----------------------------
# Step 8: Plot Confusion Matrix
# ----------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["NORMAL", "PNEUMONIA"],
    yticklabels=["NORMAL", "PNEUMONIA"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Pneumonia Detection")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print(" Confusion matrix saved as confusion_matrix.png")
