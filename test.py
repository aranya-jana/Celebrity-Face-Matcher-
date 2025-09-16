import cv2
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Load features and filenames
feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

# Load VGGFace model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

# Initialize MTCNN
detector = MTCNN()

# Load sample image and convert to RGB
sample_img = cv2.imread('sample/r_sing.png')
sample_img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

# Detect faces
results = detector.detect_faces(sample_img_rgb)
if len(results) == 0:
    raise Exception("No face detected")

# Extract face coordinates
x, y, width, height = results[0]['box']
x, y = max(0, x), max(0, y)  # ensure coordinates are non-negative
face = sample_img_rgb[y:y+height, x:x+width]

# Resize and preprocess face for VGGFace
face_img = Image.fromarray(face).resize((224,224))
face_array = np.asarray(face_img).astype('float32')
expanded_img = np.expand_dims(face_array, axis=0)
preprocessed_img = preprocess_input(expanded_img)

# Extract features
result = model.predict(preprocessed_img).flatten()

# Compute cosine similarity with all features
similarity = [cosine_similarity(result.reshape(1,-1), f.reshape(1,-1))[0][0] for f in feature_list]

# Find the most similar face
index_pos = np.argmax(similarity)

# Show recommended image
temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('output', temp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
