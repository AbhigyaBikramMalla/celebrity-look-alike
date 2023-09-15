
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
from PIL import Image


features_list=np.array(pickle.load(open('embedding.pkl','rb')))
filenames=pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
detector=MTCNN()

sample_img=cv2.imread('sample/shah.jpg')
results=detector.detect_faces(sample_img)

x,y,width,height=results[0]['box']

face=sample_img[y:y+height,x:x+width]

#extract features loaded image in the top


image=Image.fromarray(face)
image=image.resize((224,224))

face_array=np.asarray(image)
face_array=face_array.astype('float32')
expanded_img=np.expand_dims(face_array,axis=0)
preprocessed_img=preprocess_input(expanded_img)

result=model.predict(preprocessed_img)

#find cosine similarity
similarity=[]
for i in range(len(features_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1),features_list[i].reshape(1,-1))[0][0])




#recommend highest cosine similarity
#indexing value in similarity
index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
temp_img=cv2.imread(filenames[index_pos])
cv2.imshow('output',temp_img)
cv2.waitKey(0)
