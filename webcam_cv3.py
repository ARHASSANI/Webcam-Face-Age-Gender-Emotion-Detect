import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
from keras.models import load_model
import os

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

i=1
Path='./frame.jpg'
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #video_capture.read()
   

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
        

            

    
    if cv2.waitKey(1) & 0xFF == ord('q'):#quite
        if os.path.isfile(Path)==True:
            os.remove(Path)
        break


        


    

    # age prediction 
    def ages(blobs):
        ageProto = "age_deploy.prototxt"
        ageModel = "age_net.caffemodel"
        ageNet = cv2.dnn.readNet(ageModel, ageProto)

        ageList = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']
        ageNet.setInput(blobs)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        # print("Age Output : {}".format(agePreds))
        #print("Age : {}".format(age))
        return age


    #Gender Prediction
    def Gender(blobs):
        genderProto = "gender_deploy.prototxt"
        genderModel = "gender_net.caffemodel"
        genderNet= cv2.dnn.readNet(genderModel, genderProto)
        genderList = ['Male', 'Female']
        genderNet.setInput(blobs)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        # print("Gender Output : {}".format(genderPreds))
        #print("Gender : {}".format(gender))
        return gender

    #Display Age  and Gender
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    blob = cv2.dnn.blobFromImage(frame, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    age=ages(blob)
    gender= Gender(blob)

    label = "{}, {}".format(gender, age)
    for (x, y, w, h) in faces:
        cv2.putText(frame, label, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3, cv2.LINE_AA)
    
    #Display emotion
    def Emotion(paths):
        emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
        model =load_model("model_v6_23.hdf5")
        face_image  = cv2.imread(paths)
        face_image = cv2.resize(face_image, (48,48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
        predicted_class = np.argmax(model.predict(face_image))
        label_map = dict((v,k) for k,v in emotion_dict.items())
        predicted_label = label_map[predicted_class]

        print(label,predicted_label)
        return predicted_label

 
   
    if cv2.waitKey(1) & 0xFF == ord('c'):#Capture
        A=cv2.imwrite(Path.format(i),frame)
    

        if A==True:
            try:
                img = cv2.imread(Path,cv2.COLOR_BGR2GRAY)
             
                Emotion_label=Emotion(Path)
                
                cv2.putText(img, Emotion_label, (11, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3, cv2.LINE_AA)

                cv2.imshow("image", img)

                
            except:
                print("Error")
                


    # Display the resulting frame
    cv2.imshow('Video', frame)

   

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
