import os
import pickle
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="Bollywood Celebrity Matcher",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar instructions
with st.sidebar:
    st.title("How to Use")
    st.markdown("""
    - **Step 1**: Upload a clear, front-facing photo.
    - **Step 2**: Wait a few seconds for processing.
    - **Step 3**: See which Bollywood celebrity you resemble!
    """)
    st.info("No personal data is stored. Images are processed locally.")

# Ensure uploads folder exists
os.makedirs('uploads', exist_ok=True)

# Load model and features
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')
feature_list = pickle.load(open('embedding.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))

# Save uploaded image
def save_uploaded_image(uploaded_image):
    try:
        path = os.path.join('uploads', uploaded_image.name)
        with open(path,'wb') as f:
            f.write(uploaded_image.getbuffer())
        return path
    except Exception:
        return None

# Extract facial features (224x224 for model)
def extract_features(img_path, model, detector):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img)
    if not results:
        raise Exception("No face detected in the image.")
    x, y, width, height = results[0]['box']
    x, y = max(0, x), max(0, y)
    face = img[y:y+height, x:x+width]
    face_img = Image.fromarray(face).resize((224,224))  # Model requires 224x224
    face_array = np.asarray(face_img).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    return model.predict(preprocessed_img).flatten()

# Recommend celebrity
def recommend(feature_list, features):
    similarity = [cosine_similarity(features.reshape(1,-1), f.reshape(1,-1))[0][0] for f in feature_list]
    return np.argmax(similarity), max(similarity)

# Extract celebrity name from filename
def get_celebrity_name(file_path):
    base = os.path.basename(file_path)
    name = base.split('.')[0].replace('_', ' ')
    return name

# Main UI
st.title("🎬 Bollywood Celebrity Matcher")
st.markdown(
    "<h4 style='color: #6c63ff; text-align:center;'>Find out which Bollywood celebrity you resemble using AI face recognition!</h4>",
    unsafe_allow_html=True
)

uploaded_image = st.file_uploader("Upload your photo (jpg/jpeg/png)", type=['jpg','jpeg','png'])

if uploaded_image is not None:
    image_path = save_uploaded_image(uploaded_image)
    if image_path:
        display_image = Image.open(uploaded_image)
        try:
            features = extract_features(image_path, model, detector)
            index_pos, similarity_score = recommend(feature_list, features)
            predicted_actor = get_celebrity_name(filenames[index_pos])

            # Two centered containers for images side by side, same box size
            container = st.container()
            with container:
                col1, col2, col3 = st.columns([1,1,1])
                with col1:
                    st.write("")  # empty spacing
                with col2:
                    st.image(display_image.resize((150,150)), width=150)  # same size
                with col3:
                    st.image(Image.open(filenames[index_pos]).resize((150,150)), width=150)

            # "You look like" plain centered text
            st.markdown(
                f"<h3 style='text-align:center; color:#4B0082;'>You look like <b>{predicted_actor}</b>!</h3>",
                unsafe_allow_html=True
            )

            # Similarity in transparent green box
            st.markdown(
                f"""
                <div style="background-color: rgba(76, 175, 80, 0.3); padding:10px; border-radius:8px; text-align:center; width:200px; margin:auto;">
                    <p style="color:white; margin:0;">Similarity: {similarity_score*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Failed to save the uploaded image. Please try again.")

# Footer
st.markdown(
    "<hr><center><small>Powered by VGGFace & MTCNN | For entertainment purposes only.</small></center>",
    unsafe_allow_html=True
)
