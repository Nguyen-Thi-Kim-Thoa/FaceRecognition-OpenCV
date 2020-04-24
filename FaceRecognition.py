# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:47:25 2020

@author: trung98
"""


import cv2 
import os
import numpy as np

subjects =['','Nhat Trung','T.Nguyen Xuan Phuc','Hulk','Join Sena','Jhon Wick','Halle Berry'
           ,'Seguri','G Dragon','Top','Taeyang','Daesung']
face_cascade = cv2.CascadeClassifier('OpenCV-Files/lbpcascade_frontalface.xml')
#smile_cascage = cv2.CascadeClassifier('OpenCV-Files/.xml')
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor = 1.1 ,
        minNeighbors = 5 ,
        )
    if len(faces)==0 :
        return None,None
    (x, y, w, h) = faces[0]
    
    return gray[y:y+w, x:x+h], faces[0]

def detect_faces(img):
    grays = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor = 1.1 , 
        minNeighbors = 5 
        )
    if len(faces) == 0 :
        return None , None
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        grays.append(roi_gray)
    return grays , faces

def detect_faces_camera(camera):
    camera.set(3,1000)
    camera.set(4,600)
    while True:
        ret, img =camera.read()
    return img , detect_faces(img)
        
def prepare_training_data(data_folder_path):
    
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    for dir_name in dirs :
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name 
        #get the images names that are inside the given subject diretory
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
            face, rect = detect_face(image)
            if face is not None:    
                faces.append(face)
                labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels
 
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
    
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
def predict(test_img):
    img = np.copy(test_img)
    face , rect = detect_face(img)
    label , confidence = face_recognizer.predict(face)
    if confidence < 100 :
        print(confidence)      
        confidence = " {0}%".format(round(100-confidence))
        label_text = subjects[label] + confidence
    else:
        label_text = "unknown"
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
    return img

def predict2(test_img):
    img = np.copy(test_img)
    faces , rects = detect_faces(img)
    i = 0
    for (x,y,w,h) in rects: 
        rect = (x,y,w,h)
        label , confidence = face_recognizer.predict(faces[i])
        i+=1
        if confidence < 100 : 
         confidence = " {0}%".format(round(100-confidence))
         label_text = subjects[label] + confidence
        else:
         label_text = "unknown"
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
        draw_text(img, label_text, x, y-5)
    return img

def predict3():
    print("Open Camera")
    cam = cv2.VideoCapture(0)
    cam.set(3, 1000) # set video widht
    cam.set(4, 600) # set video height 
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    while True:
       ret, img = cam.read()
       gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
       faces = face_cascade.detectMultiScale(gray,
                                             scaleFactor = 1.1 , 
                                             minNeighbors = 5 ,
                                             )
       for (x,y,w,h) in faces:
           rect = (x,y,w,h)
           label, confidence = face_recognizer.predict(gray[y:y+h,x:x+w])        
           if (confidence < 100):
             confidence = "  {0}%".format(round(100 - confidence))
             label_text = subjects[label] + confidence
           else:
             label_text = "unknown"
           draw_rectangle(img, rect)
           draw_text(img, label_text, x, y-5)  
       cv2.imshow('camera',img) 
       k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
       if k == 27:
          break
    cam.release()
    cv2.destroyAllWindows()
#load test images
print("Preparing predict")
predict3()
'''
img1 = cv2.imread("test-data/test2.jpg")
img2 = cv2.imread("test-data/test3.jpg")
img3 = cv2.imread("test-data/test6.jpg")
img4 = cv2.imread("test-data/test5.jpg")
#perform a prediction
predicted_img1 = predict2(img1)
predicted_img2 = predict2(img2)
predicted_img3 = predict2(img3)
predicted_img4 = predict2(img4)


cv2.imshow("image1", predicted_img1)
cv2.imshow("image2", predicted_img2)
cv2.imshow("image3", predicted_img3)
cv2.imshow("image4", predicted_img4)    
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
'''

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        