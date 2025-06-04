from flask import Flask, request, jsonify, render_template  # Added render_template
import cv2
import numpy as np
import base64
import mediapipe as mp
import pandas as pd
import copy
import itertools
import string
from tensorflow import keras

app = Flask(__name__)

# Load model
model = keras.models.load_model("static/model.h5")

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5  # Fixed spelling
)

@app.route('/')
def index():
    return render_template('index.html')  # Changed to use templates


alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

# Utility functions
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index in range(len(temp_landmark_list)):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    return [x / max_value if max_value != 0 else 0 for x in temp_landmark_list]



@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        decoded = base64.b64decode(image_data)
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return "None"

        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_list = calc_landmark_list(img, hand_landmarks)
        processed = pre_process_landmark(landmark_list)
        df = pd.DataFrame([processed])

        prediction = model.predict(df, verbose=0)
        predicted_class = np.argmax(prediction)
        label = alphabet[predicted_class]
        return label
    except Exception as e:
        print(e)
        return "Error"

if __name__ == '__main__':
    app.run(debug=True)
