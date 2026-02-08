import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# CBAM Attention Block
def cbam_block(feature_map, ratio=8):

    channel = feature_map.shape[-1]

    shared_layer_one = Dense(channel//ratio,
                             activation='relu')
    shared_layer_two = Dense(channel)

    avg_pool = GlobalAveragePooling2D()(feature_map)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(feature_map)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    attention = Add()([avg_pool, max_pool])
    attention = Activation('sigmoid')(attention)

    attention = Multiply()([feature_map, attention])

    return attention


def build_model():

    inputs = Input(shape=(224,224,3))

    x = Conv2D(32, (3,3), activation="relu")(inputs)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3,3), activation="relu")(x)
    x = MaxPooling2D()(x)

    # Attention Block
    x = cbam_block(x)

    x = Conv2D(128, (3,3), activation="relu")(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
