import cv2, os, numpy as np
from PIL import Image

cam = cv2.VideoCapture(0)  # Membuat variable kamera dengan camera 1
cam.set(3, 450) # Mengubah Lebar cam
cam.set(4,480) # Mengubah Tinggi cam

wajahDir = 'faces-data'
latihDir = 'faces-train'

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Lib deteksi wajah
eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml') # Lib deteksi mata

# ------------------------------------------------------------------------------------
# Capture Faces
# ------------------------------------------------------------------------------------

faceID = input("Masukkan Face ID yang akan direkam [1-100]: ")
print ("Tatap Wajah kedalam webcam dengan posisi sesuai, Tunggu proses pengambilan data wajah selesai...")

ambilData = 1

while True:
    retV, frame = cam.read()  # Membaca Camera
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Membuat Warna Grey
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y),(x+w,y+h),(0,0,255),2) # Frame Warna Merah

        namaFile = 'faces-from.'+str(faceID)+'.'+str(ambilData)+'.jpg'
        cv2.imwrite(wajahDir+'/'+namaFile,frame)
        ambilData +=1

        roiAbuAbu = abuAbu[y:y+h,x:x+w]
        roiWarna = frame[y:y+h,x:x+w]
        eyes = eyeDetector.detectMultiScale(roiAbuAbu)
        for (xe,ye,we,he) in eyes:
            cv2.rectangle(roiWarna,(xe,ye),(xe+we,ye+he),(255,0,0),1)

    cv2.imshow('WebRecord', frame)  # Memunculkan Windows3
    k = cv2.waitKey(1) & 0xFF  # Menunggu tombol ditekan untuk keluar
    if ambilData>30:
        print("Perekaman telah berhasil, melanjutkan proses train data")
        break

cam.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------------------------------
# Train The AI from Face Recog
# ------------------------------------------------------------------------------------

def getImageLabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    faceIDs = []
    for imagePath in imagePaths:
        PILImg = Image.open(imagePath).convert('L')  # Convert kedalam grey
        imgNum = np.array(PILImg, 'uint8')
        faceID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = faceDetector.detectMultiScale(imgNum)
        for (x, y, w, h) in faces:
            faceSamples.append(imgNum[y:y + h, x:x + w])
            faceIDs.append(faceID)
    return faceSamples, faceIDs


faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

print("Mesin sedang melakukan training data wajah, tunggu beberapa saat")
faces, IDs = getImageLabel(wajahDir)
faceRecognizer.train(faces, np.array(IDs))

# Simpan
faceRecognizer.write(latihDir + '/training.xml')
#faceRecognizer.write(latihDir + '/training.yml')
print("sebanyak {0} data wajah telah di traingkan ke mesin.".format(len(np.unique(IDs))))
