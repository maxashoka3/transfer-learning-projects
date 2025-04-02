import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("model/xray_pneumonia_model.h5")

# Class labels
labels = ["Normal", "Pneumonia"]

def predict_pneumonia(img):
    img = img.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    result = labels[int(prediction[0] > 0.5)]  # Threshold at 0.5
    return result

# Create Gradio Interface
iface = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Pneumonia Detection",
    description="Upload a Chest X-ray to detect Pneumonia"
)

# Run the app
if __name__ == "__main__":
    iface.launch()
