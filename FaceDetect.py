
import cv2, os, numpy as np

wajahDir = 'faces-data'
latihDir = 'faces-train'

cam = cv2.VideoCapture(0)  # Membuat variable kamera dengan camera 1
cam.set(3, 450) # Mengubah Lebar cam
cam.set(4,480) # Mengubah Tinggi cam

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Lib deteksi wajah
eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml') # Lib deteksi mata
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.read(latihDir+'/training.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Tidak Dikethaui', 'Rivaldi', 'Mufiz', 'Tresna', 'ilham', 'Imad']

minWidh = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

while True:
    retV, frame = cam.read()  # Membaca Camera
    frame = cv2.flip(frame, 1)
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Membuat Warna Grey
    faces = faceDetector.detectMultiScale(abuAbu, 1.2, 5,minSize=(round(minWidh),round(minHeight)))
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y),(x+w,y+h),(0,255,0),1) # Frame Warna Merah

        roiAbuAbu = abuAbu[y:y+h,x:x+w]
        roiWarna = frame[y:y+h,x:x+w]
        eyes = eyeDetector.detectMultiScale(roiAbuAbu)
        for (xe,ye,we,he) in eyes:
            cv2.rectangle(roiWarna,(xe,ye),(xe+we,ye+he),(255,0,0),1)

            id, confidence = faceRecognizer.predict(abuAbu[y:y+h,x:x+w])
            if confidence <= 60 :
                nameID = names[id]
                confidenceTxt = " {0}%".format(round(100-confidence))
            else:
                nameID = names[0]
                confidenceTxt = " {0}%".format(round(100-confidence))

        cv2.putText(frame,str(nameID),(x+5,y-5),font,1,(255,255,255),2)
        cv2.putText(frame,str(confidenceTxt),(x+5,y+h-5),font,1,(255,255,0),1)

    cv2.imshow('Recognition Wajah', frame)  # Memunculkan Windows
    k = cv2.waitKey(1) & 0xFF  # Menunggu tombol ditekan untuk keluar
    if k == ord('q') or k == 27: # Tombol Q dan ESC untuk keluar
        break

print("EXIT")
cam.release()
cv2.destroyAllWindows()