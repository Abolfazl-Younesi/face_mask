import cv2
import time
import numpy as np
from keras.models import load_model

model = load_model("inc_fine.h5")

results = {0: 'incorrect_mask', 1: 'with_mask', 2: 'without_mask'}
GR_dict = {0: (0, 128, 255), 1: (0, 255, 0), 2: (0, 0, 255)}

rect_size = 4
cap = cv2.VideoCapture(0)

haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

while True:
    (rval, im) = cap.read()

    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        start = time.time()
        (x, y, w, h) = [v * rect_size for v in f]

        face_img = im[y:y + h, x:x + w]
        rerect_sized = cv2.resize(face_img, (299, 299))
        normalized = rerect_sized / 255.0
        reshaped = np.reshape(normalized, (1, 299, 299, 3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        fps = str(round(1.0 / (time.time() - start)))
        cv2.rectangle(im, (x, y), (x + w, y + h), GR_dict[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), GR_dict[label], -1)
        cv2.putText(im, results[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(im, str('FPS: ' + fps), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 180), 2)
    cv2.imshow('LIVE', im)
    key = cv2.waitKey(10)

    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()
