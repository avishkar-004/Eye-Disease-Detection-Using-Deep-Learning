import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    """Generate Grad-CAM heatmap for model interpretability."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image."""
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(
        np.uint8(255 * img), 1 - alpha, heatmap, alpha, 0
    )
    return superimposed


def visualize_gradcam(model, image_path, last_conv_layer, save_path=None):
    """Complete Grad-CAM visualization pipeline."""
    from PIL import Image

    img = Image.open(image_path).resize((256, 256))
    img_array = np.array(img) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    heatmap = get_gradcam_heatmap(model, input_tensor, last_conv_layer)
    overlay = overlay_gradcam(img_array, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
