# import os
import pickle

# actors = os.listdir('DataSet')

# filenames = []

# for actor in actors:
#     for file in os.listdir(os.path.join('DataSet',actor)):
#         filenames.append(os.path.join('DataSet',actor,file))

# pickle.dump(filenames,open('filenames.pkl','wb'))


import numpy as np
# from tqdm import tqdm
from tensorflow.keras.preprocessing import image                                               # type: ignore
from tensorflow.keras.utils import get_source_inputs  # patched import                         # type: ignore
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

# Load list of image file paths
filenames = pickle.load(open('filenames.pkl','rb'))

# Load VGGFace model with ResNet50 backbone
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

# print(model.summary())

