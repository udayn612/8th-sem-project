import cv2
import time
import os
import numpy as np
from PIL import Image
import pickle
import sqlite3 

recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('recognizer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
##def SendMail(ImgFileName):
##    img_data = open(ImgFileName, 'rb').read()
##    msg = MIMEMultipart()
##    msg['Subject'] = 'person'
##    msg['From'] = 'udayn612@gmail.com'
##    msg['To'] = 'gupta.yash1996@gmail.com'
##
##    text = MIMEText("test")
##    msg.attach(text)
##    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
##    msg.attach(image)
##
##    s = smtplib.SMTP('smptp.gmail.com',587)
##    s.ehlo()
##    s.starttls()
##    s.ehlo()
##    s.login("udayn612@gmail.com", "ankushn@0123")
##    s.sendmail(From, To, msg.as_string())
##    s.quit()


def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM people WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile  


cam = cv2.VideoCapture(1)
time.sleep(1)
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
while True:
    ret, im =cam.read()
    if ret == True:
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            profile=getProfile(id)
            if(profile!=None):
                cv2.cv.PutText(cv2.cv.fromarray(im),str(profile[1]), (x,y+h),font, 255)
                cv2.imwrite("dataset1/User.jpg", gray[y:y+h,x:x+w])
               ## ImgFileName1="C:\Users\udayn_000\Downloads\Programs\opencv\dataset1\User.1.capture.jpg"
                ##SendMail(ImgFileName1)
                
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
