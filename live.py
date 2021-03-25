import numpy as np
import cv2
from tensorflow import keras



model = keras.models.load_model('model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cropped_img = img[y:y + h, x:x + w]
        cropped_img = cv2.resize(cropped_img, (150, 150))
        cropped_img = np.reshape(cropped_img, [1, 150, 150, 3]) / 255

        prediction = model.predict(cropped_img)
        text = 'Mask' if prediction.argmax() == 0 else 'No Mask'

        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (110, 255, 255), 2)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

