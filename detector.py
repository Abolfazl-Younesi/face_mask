import cv2
import numpy as np
from keras.models import load_model

model = load_model("inc_fine.h5")

results = {0: 'incorrect_mask', 1: 'with_mask', 2: 'without_mask'}
GR_dict = {0: (0, 128, 255), 1: (0, 255, 0), 2: (0, 0, 255)}

rect_size = 4
cap = cv2.VideoCapture(0)

haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread('assets/dataset/test/without_mask/270.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = haarcascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:

    face_img = img[y:y + h, x:x + w]
    rerect_sized = cv2.resize(face_img, (299, 299))
    normalized = rerect_sized / 255.0
    reshaped = np.reshape(normalized, (1, 299, 299, 3))
    reshaped = np.vstack([reshaped])
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    cv2.rectangle(img, (x, y), (x + w, y + h), GR_dict[label], 2)
    cv2.rectangle(img, (x, y - 40), (x + w, y), GR_dict[label], -1)
    cv2.putText(img, results[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

cv2.imshow('image', img)
key = cv2.waitKey(0)

cap.release()

cv2.destroyAllWindows()
