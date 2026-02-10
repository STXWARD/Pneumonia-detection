import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# =========================
# Config
# =========================

IMG_HEIGHT = 150
IMG_WIDTH = 150

st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Pneumonia Detection from Chest X-Ray")
st.markdown("### AI-Powered Medical Imaging Analysis")

st.sidebar.header("About")
st.sidebar.info("""
**Model Performance:**
- Accuracy: 88.94%
- Pneumonia Detection: 97.7%

Educational demo only.
""")

# =========================
# Load model
# =========================

@st.cache_resource
def load_model():
    # Load the model
    m = tf.keras.models.load_model("model/pneumonia_detection_model.h5")
    # Force build the model by passing a dummy tensor
    m(tf.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3)))
    return m

model = load_model()

# Auto-detect the last conv layer
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

# Optional: verify model is ready
st.success(f"✅ Model loaded. Using layer: {last_conv_layer_name}")

# =========================
# Grad-CAM functions
# =========================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Keras 3 fix: Access internal layer tensors directly instead of model.input/output
    try:
        model_input = model.layers[0].input
        target_layer = model.get_layer(last_conv_layer_name).output
        classifier_output = model.layers[-1].output
    except Exception:
        # Fallback if standard access fails
        model_input = model.inputs[0]
        target_layer = model.get_layer(last_conv_layer_name).output
        classifier_output = model.output

    # Create the sub-model
    grad_model = tf.keras.Model(inputs=model_input, outputs=[target_layer, classifier_output])

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        
        # Watch the conv layer output to ensure gradients can be computed
        tape.watch(conv_outputs)
        
        # Target the prediction (binary classification index 0)
        loss = tf.math.log(predictions[:, 0] / (1 - predictions[:, 0] + 1e-10))

    # Compute gradients of the loss w.r.t. the conv layer output
    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return np.zeros((10, 10)) # Safety fallback if something goes wrong

    # Mean intensity of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.6): # Increased alpha for better visibility
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Optional: Boost the "hot" areas so they are easier to see
    heatmap = np.power(heatmap, 2) # Squaring it removes low-level "noise"
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img

# =========================
# UI
# =========================

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"] 
)

threshold = st.slider(
    "Prediction Threshold",
    0.3, 0.9, 0.7, 0.1
)

# =========================
# Main Logic
# =========================

if uploaded_file:
    col1, col2 = st.columns(2)
    show_cam = st.checkbox("Show Grad-CAM explanation")

    with col1:
        st.subheader("Uploaded X-Ray")
        # .convert("RGB") is crucial to strip Alpha channels from PNGs
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

    if st.button("🔬 Analyze X-Ray", type="primary"):
        with st.spinner("Analyzing..."):

            # -------- Preprocessing --------
            img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # -------- Prediction --------
            prediction = model.predict(img_array, verbose=0)[0][0]

            if prediction > threshold:
                result = "PNEUMONIA"
                confidence = prediction * 100
            else:
                result = "NORMAL"
                confidence = (1 - prediction) * 100

        # =========================
        # Display results
        # =========================

        with col2:
            st.subheader("Result")

            if result == "PNEUMONIA":
                st.error(f"🦠 **{result}**")
            else:
                st.success(f"✅ **{result}**")

            st.metric("Confidence", f"{confidence:.2f}%")
            st.progress(float(confidence / 100))

            # =========================
            # Grad-CAM
            # =========================

            if show_cam:
                heatmap = make_gradcam_heatmap(
                    img_array,
                    model,
                    last_conv_layer_name
                )

                # Use the already RGB-converted image for the overlay base
                original_np = np.array(img_resized)

                cam_image = overlay_heatmap(
                    original_np,
                    heatmap
                )

                st.subheader("Grad-CAM Explanation")
                st.image(
                    cam_image,
                    caption="AI Focus Area (Red = High Importance)",
                    channels="RGB"
                )

            # =========================
            # Debug info
            # =========================

            with st.expander("Technical Details"):
                st.write(f"Raw prediction: {prediction:.4f}")
                st.write(f"Threshold used: {threshold}")
                st.write(f"Input shape: {img_array.shape}")

# =========================
# Footer
# =========================

st.markdown("---")
st.markdown("""
**AI-powered Pneumonia Detection System**  
Developed by Steward Jacob  
🔗 http://www.linkedin.com/in/stewardjacob
""")