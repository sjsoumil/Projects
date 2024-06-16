import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = "ImagesAttendance"
images = []
ClassNames = []
# Providing the path for images
myList = os.listdir(r"C:\Users\91825\Desktop\Computer Vision]\Face Recognition\Images")

# Loop for getting images and names of the images from the path
for x in myList:
    curImg = cv2.imread(r"C:\Users\91825\Desktop\Computer Vision]\Face Recognition\Images" + '/' + x)
    images.append(curImg)
    ClassNames.append(os.path.splitext(x)[0])

# Function to convert and encode the images and return the encoded list
def findencodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Ensure at least one encoding is found
            encodeList.append(encode[0])
    return encodeList


def MarkAttendance(name):
    with open(r"C:\Users\91825\Desktop\Computer Vision]\Face Recognition\Project\Attendance.csv", 'r+') as f:
    # Your file operations here

        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dtString}')

        print(myDataList)




encodeListKnown = findencodings(images)
print("Encoding done")

# Initialize the web cam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        continue

    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        if faceDis.size > 0:  # Ensure face distances are not empty
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = ClassNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                MarkAttendance(name)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
