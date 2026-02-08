import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def grad_cam(model, img_array, layer_name="conv2d_2"):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


def show_gradcam(img_path, model):

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img_array = np.expand_dims(img/255.0, axis=0)

    heatmap = grad_cam(model, img_array)

    plt.imshow(img)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.title("Grad-CAM Explanation")
    plt.axis("off")
    plt.show()
