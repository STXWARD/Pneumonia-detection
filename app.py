import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# CRITICAL: These must match your training configuration
IMG_HEIGHT = 150
IMG_WIDTH = 150

st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🫁", layout="centered")

st.title("🫁 Pneumonia Detection from Chest X-Ray")
st.markdown("### AI-Powered Medical Imaging Analysis")

st.sidebar.header("About")
st.sidebar.info("""
**Model Performance:**
- Accuracy: 88.94%
- Pneumonia Detection: 97.7%

**Note:** Educational demonstration tool.
Always consult medical professionals.
""")

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Load model - make sure this file is in the same folder!
        model = tf.keras.models.load_model("model/pneumonia_detection_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model()

if model is None:
    st.error("Cannot proceed without model. Make sure pneumonia_detection_model.keras is in the same folder.")
    st.stop()
else:
    st.success(f"✅ Model loaded! Expected input: {IMG_WIDTH}x{IMG_HEIGHT}")

# File uploader
uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])

# Threshold slider
threshold = st.slider("Prediction Threshold", 0.3, 0.9, 0.7, 0.1)

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded X-Ray")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    if st.button("🔬 Analyze X-Ray", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                # CRITICAL PREPROCESSING - MUST MATCH TRAINING
                # Convert to RGB
                img = image.convert('RGB')
                
                # Resize to EXACT training dimensions
                img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                
                # Convert to numpy array
                img_array = np.array(img, dtype=np.float32)
                
                # Normalize to [0, 1]
                img_array = img_array / 255.0
                
                # Add batch dimension
                img_array = np.expand_dims(img_array, axis=0)
                
                # Verify shape
                expected_shape = (1, IMG_HEIGHT, IMG_WIDTH, 3)
                if img_array.shape != expected_shape:
                    st.error(f"Shape error: got {img_array.shape}, expected {expected_shape}")
                    st.stop()
                
                # Make prediction
                prediction = model.predict(img_array, verbose=0)[0][0]
                
                # Apply threshold
                if prediction > threshold:
                    result = "PNEUMONIA"
                    confidence = prediction * 100
                else:
                    result = "NORMAL"
                    confidence = (1 - prediction) * 100
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)
                st.stop()
        
        # Display results
        with col2:
            st.subheader("Result")
            
            if result == "PNEUMONIA":
                st.error(f"🦠 **{result}**")
            else:
                st.success(f"✅ **{result}**")
            
            st.metric("Confidence", f"{confidence:.2f}%")
            st.progress(float(confidence / 100))
            
            with st.expander("Technical Details"):
                st.write(f"Raw prediction: {prediction:.4f}")
                st.write(f"Threshold used: {threshold}")
                st.write(f"Input shape: {img_array.shape}")

st.markdown("---")
st.markdown("""
**AI-powered Pneumonia Detection System**  
Developed by Steward Jacob  
🔗 [LinkedIn](http://www.linkedin.com/in/stewardjacob)
""")

