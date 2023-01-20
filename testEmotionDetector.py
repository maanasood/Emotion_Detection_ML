import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear',
                3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

json_file = open('Emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights('Emotion_model.h5')
print("Loaded model from disk")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                          'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(
        gray_frame, 1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y+h, x: x+w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxIndex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxIndex], (x+5, y-20),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f'Emotion generated is:{emotion_dict[maxIndex]}')
