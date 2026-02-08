import os
from tensorflow.keras.callbacks import EarlyStopping


# Step 1: Local Dataset Path
DATA_DIR = "chest_xray"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

print("Train folder:", TRAIN_DIR)
print("Validation folder:", VAL_DIR)
print("Test folder:", TEST_DIR)

# Step 2: Load dataset
from dataset_loader import load_data
train_data, val_data = load_data(TRAIN_DIR, VAL_DIR)

# Step 3: Build model
from attention_model import build_model
model = build_model()

# Step 4: Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
)


model.save("attention_pneumonia_model.h5")
print("✅ Model Saved Successfully!")


