import streamlit as st
from keras.models import load_model 
import cv2 
import numpy as np

st.set_page_config(page_title="Mask Detection System",page_icon="https://i.pinimg.com/736x/5b/a8/cc/5ba8cc62355c908fb3965aa728daa814.jpg")
st.title("Face Mask Detection System")
choice=st.sidebar.selectbox("My menu",("Home","IP Camera","Camera","Show Results"))

if(choice=="Home"):
    st.image("https://act-my.com/wp-content/uploads/2021/12/faceRecognition.gif")
    st.write("This application is regarding detecting the peoples who are wearing mask and without mask.")
elif(choice == "IP Camera"):
    url=st.text_input("Enter IP camer URL: ")
    window=st.empty() #here we are trying to create a empty component so that we can show images later on
    #it is same like window we showed in opencv output the video that gets on people wearing nd not weraing mask like tht.
    btn=st.button("Start Detection")
    if(btn):
        vid=cv2.VideoCapture(url)
        facemodel=cv2.CascadeClassifier("face.xml")
        maskmodel=load_model("mask.h5",compile=False)
        btn2=st.button("Stop Detection")
        if(btn2):
            vid.release() #here we release the video bcz we dont need it further
            st.experimental_rerun() #this function will reload everything.means it will rerun the entire function so itll go to the initial stage.initial stage means everything will be gone.
        i=1

        while True:
            flag,frame=vid.read()
            if flag:
                faces=facemodel.detectMultiScale(frame)        
                for (x,y,l,w) in faces:
                    face_img=frame[y:y+w,x:x+l] 
                    face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)
                    face_img = np.asarray(face_img, dtype=np.float32).reshape(1, 224, 224, 3)
                    face_img = (face_img / 127.5) - 1
           
                    pred=maskmodel.predict(face_img)[0][0] 
           
                    if(pred>0.9):
                        path="mydata/"+str(i)+".png" #this will save in  the format 1.png,2.png,3.png etc
                        cv2.imwrite(path,frame[y:y+w,x:x+l]) #frame[y:y+w,x:x+l] - to save the correct cropped face
                        i=i+1
                        #from path to i above three lines will run oly if the person not wearing mask.
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
                window.image(frame,channels="BGR") #it is the channels of open cv and it should be in same color code order.otherwise it will get disturbed and proper color coding will not be done.
#window variable is a emply streamlit component,now you can pass any component to it,so here we will pass image component.
            

elif(choice == "Camera"):
    cam=st.selectbox("Choose Camera",("None","Primary","Secondary"))
    window=st.empty() 
    btn=st.button("Start Detection")
    if(btn):
        c=0
        if(cam=="Secondary"):
            c=1
        vid=cv2.VideoCapture(c)
        facemodel=cv2.CascadeClassifier("face.xml")
        maskmodel=load_model("mask.h5",compile=False)
        btn2=st.button("Stop Detection")
        if(btn2):
            vid.release() 
            st.experimental_rerun() 
        i=1

        while True:
            flag,frame=vid.read()
            if flag:
                faces=facemodel.detectMultiScale(frame)        
                for (x,y,l,w) in faces:
                    face_img=frame[y:y+w,x:x+l] 
                    face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)
                    face_img = np.asarray(face_img, dtype=np.float32).reshape(1, 224, 224, 3)
                    face_img = (face_img / 127.5) - 1
           
                    pred=maskmodel.predict(face_img)[0][0] 
           
                    if(pred>0.9):
                        path="mydata/"+str(i)+".png" 
                        cv2.imwrite(path,frame[y:y+w,x:x+l]) 
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
                window.image(frame,channels="BGR")

elif(choice=="Show Results"):
    if "h" not in st.session_state:
        st.session_state["h"]=1
    btn4=st.button("Next Image")
    if btn4:
        st.session_state["h"]=st.session_state["h"]+1
    btn5=st.button("Previous Image")
    if btn5:
        st.session_state["h"]=st.session_state["h"]-1        
    st.image("mydata/"+str(st.session_state["h"])+".png")

#note: if u hv usb camera(it is a camera where you can connect it with the help of usb or any kind
#of camera having wire)and if u connect your usb to ur lapy to ur desktop,or if you have webcam already
#in your lapy and also u r also connecting ur camera with the help of usb to ur lapy,then your webcam is considered as 0
#and ur usb camera is considered as 1.

#And you consider u hv desktop and you dont have internal camera or webcam in it and if you connect
#your usb camera to it then in that case the the usb camera will be 0

#0-->primary camera
#1-->secondary camera




















                        
    
