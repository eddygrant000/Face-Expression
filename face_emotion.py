import cv2
import imutils
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import random, sys, time

old_data = dict()
starting_time = time.time()

detection_file_model = 'haarcascades/haarcascade_frontalface_default.xml'
allemotions = ["angry" ,"disgust", "scared", "happy", "sad", "surprised", "neutral"]
detect_face = cv2.CascadeClassifier(detection_file_model)
#emotion_file_model = 'models/_mini_XCEPTION.106-0.65.hdf5'
emotion_file_model = 'models/_mini_XCEPTION.106-0.65.hdf5'
classifier_emoation = load_model(emotion_file_model, compile=False)
cv2.namedWindow('Capture Your Face')
camera = cv2.VideoCapture(0)

def rgb_color():
    rgbl = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    return tuple(rgbl)

def numpy_zero_create():
    return np.zeros((250, 300, 3), dtype="uint8")

def get_list_from_zip(data=None):
    global starting_time
    value_data = [value * 100 for i, value in data]
    high_value = max(value_data)
    name = allemotions[value_data.index(high_value)]
    if len(list(old_data.keys())) < 1:
        old_data[allemotions[value_data.index(high_value)]] = max(value_data)
    elif name not in list(old_data.keys()) and high_value != old_data[list(old_data.keys())[0]]:
        real_time_to_change = round(time.time() - starting_time)
        t_min, t_sec = divmod(real_time_to_change, 60)
        old_data[name] = old_data.pop(list(old_data.keys())[0])
        old_data[list(old_data.keys())[0]] = high_value
        print(old_data, t_sec)
        starting_time = time.time()

length_value = []
rbg_c = []

while True:
    frame = camera.read()[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_face.detectMultiScale(gray, 1.3, 5)
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    if len(rbg_c) == 0:
        rbg_c.append(rgb_color())
    for (x, y, w, h) in faces:
        a = faces.tolist()[:len(faces)]
        index_value = a.index([x, y, w, h])
        length_value.append(index_value + 1)
        if len(a) != len(length_value):
            rbg_c.append(rgb_color())
        cv2.rectangle(frame,(x,y),(x+w,y+h),rbg_c[index_value],2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi_gray, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = classifier_emoation.predict(roi)[0]
        np.max(preds)
        label = allemotions[preds.argmax()]
        cv2.putText(frame, "Person {}".format(index_value + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.putText(frame, label, (x + 100, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        get_list_from_zip(zip(allemotions, preds))
        for (i, (emotion, prob)) in enumerate(zip(allemotions, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)
        cv2.imshow("Person {}".format(index_value + 1), canvas)
        if index_value < len(faces):
            cv2.destroyWindow('Person {}'.format(index_value + len(faces) + 1))
    cv2.imshow('Capture Your Face', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
