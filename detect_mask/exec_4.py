# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model(args["model"])

# Para a execução
labels_dict={0:'with_mask',1:'without_mask',2:'wrong_use'}
color_dict={0:(0,255,0),1:(0,0,255),2:(255,0,0)}

size = 4
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(224,224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,224,224,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        
        #print(result)
        print('------------ inferências/precisão -----------')
        #print(result[0][0]*100)
        print(labels_dict[0]+': {:.3f}%'.format(result[0][0]*100))
        #print(result[0][1]*100)
        print(labels_dict[1]+': {:.3f}%'.format(result[0][1]*100))
        #print(result[0][2]*100)
        print(labels_dict[2]+': {:.3f}%'.format(result[0][2]*100))
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        ## coloca a legenda geral 
        '''
        cv2.rectangle(im,(2,2),(200,70),(255,255,255),-1)
        #cv2.putText(im, 'Teste 1: '+result[0], (2,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
        font                = cv2.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText = (8,20)
        fontScale           = 0.5
        fontColor           = (255,255,255)
        fontColor           = color_dict[0]
        lineType            = 1
        text                = 'Teste 1: ' + str(result[0])
        text                = labels_dict[0]+': ' + str('{:.3f}%'.format(result[0][0]*100))
        cv2.putText(im,text,topLeftCornerOfText, font, fontScale,fontColor,lineType,cv2.LINE_AA)
        text                = labels_dict[1]+': ' + str('{:.3f}%'.format(result[0][1]*100))
        cv2.putText(im,text,(8,40), font, fontScale,color_dict[1],lineType,cv2.LINE_AA)
        text                = labels_dict[2]+': ' + str('{:.3f}%'.format(result[0][2]*100))
        cv2.putText(im,text,(8,60), font, fontScale,color_dict[2],lineType,cv2.LINE_AA)
        '''
        
        # coloca a legenda para cada face
        xx = x
        yy = y-101
        # coisa
        cv2.rectangle(im,(xx,yy),(xx+160,yy+60),(255,255,255),-1)
        #cv2.putText(im, 'Teste 1: '+result[0], (2,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
        font                = cv2.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText = (2+xx,14+yy)
        fontScale           = 0.4
        fontColor           = (255,255,255)
        fontColor           = color_dict[0]
        lineType            = 1
        text                = 'Teste 1: ' + str(result[0])
        text                = labels_dict[0]+': ' + str('{:.3f}%'.format(result[0][0]*100))
        cv2.putText(im,text,topLeftCornerOfText, font, fontScale,fontColor,lineType,cv2.LINE_AA)
        text                = labels_dict[1]+': ' + str('{:.3f}%'.format(result[0][1]*100))
        cv2.putText(im,text,(2+xx,34+yy), font, fontScale,color_dict[1],lineType,cv2.LINE_AA)
        text                = labels_dict[2]+': ' + str('{:.3f}%'.format(result[0][2]*100))
        cv2.putText(im,text,(2+xx,54+yy), font, fontScale,color_dict[2],lineType,cv2.LINE_AA)
        # cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        
    # Show the image
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()