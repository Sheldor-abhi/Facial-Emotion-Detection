# import os
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
model = keras.models.load_model('Mobile_4e_7145.h5')
# model = keras.models.load_model('Mobile_4e_67.h5')
cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# print(model.summary())
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=frame[y:y+w,x:x+h] #cropping region of interest 
        roi_gray=cv2.resize(roi_gray,(224,224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'happy', 'surprise')
        # emotions=('angry','happy','neutral','surprise')
        predicted_emotion = emotions[max_index]

        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow('Display',frame)
    if cv2.waitKey(1)== ord('x'):
        cv2.destroyAllWindows()
        break

