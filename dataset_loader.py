import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocessing import apply_clahe

IMG_SIZE = 224
BATCH_SIZE = 16

def custom_preprocess(img):
    img = apply_clahe(img)
    img = img / 255.0
    return img

def load_data(train_dir, val_dir):

    datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocess,
        rotation_range=10,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    val_data = datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    return train_data, val_data
