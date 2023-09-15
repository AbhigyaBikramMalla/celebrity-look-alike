import os
import pickle
from tqdm import tqdm

import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize the ResNet-50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Specify the path to the 'image' folder


# List all image files in the 'image' folder
filenames = pickle.load(open('filenames.pkl','rb'))

# Initialize an empty list to store extracted features
features_list = []

# Define a function to extract features from an image file
def extract_features(filename):
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Extract features using the ResNet-50 model
    features = model.predict(img)
    return features

# Iterate over each image filename and extract features
for file in tqdm(filenames):
    features = extract_features(file)
    features_list.append(features)

pickle.dump(features_list,open('embedding.pkl','wb'))
# Now, features_list contains the extracted features for each image in the 'image' folder using ResNet-50
