import cv2
import joblib
import json
import pickle
import numpy as np
import base64
from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}
__model = None

def get_base64_image():
    with open('base64.txt') as f:
        return f.read()

def load_saved_artifacts():
    global __class_name_to_number
    global __class_number_to_name
    global __model

    with open("./artifacts/class_dictionary.json", 'r') as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    if __model is None:
        with open("./artifacts/saved_model.pkl", 'rb') as f:
            __model = joblib.load(f)

def classify_image(image_base64, file_path=None):
    imgs = get_cropped_img_if_2_eyes(file_path, image_base64)
    result = []
    for img in imgs:
        scaled_raw_img = cv2.resize(img, (32, 32))
        im_har = w2d(scaled_raw_img, 'db1', 5)
        scaled_img_har = cv2.resize(im_har, (32, 32))
        combined_img = np.vstack((scaled_raw_img.reshape(32 * 32 * 3, 1), scaled_img_har.reshape(32 * 32, 1)))
        len_img_arr = 32 * 32 * 3 + 32 * 32
        final = combined_img.reshape(1, len_img_arr).astype(float)
        result.append({
            'class' : __class_number_to_name[__model.predict(final)[0]],
            'probability' : np.round(__model.predict_proba(final)*100, 2).tolist()[0],
            'class_dictionary' : __class_name_to_number
        })
    return result

# function taken from stack overflow
def get_cv2_img_from_base64(base64_img):
    encoded_data = base64_img.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_img_if_2_eyes(img_path, img_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
    if img_path:
        img = cv2.imread(img_path)
    else:
        img = get_cv2_img_from_base64(img_base64_data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces

if __name__ == "__main__":
    load_saved_artifacts()
    print(classify_image(get_base64_image(),None))