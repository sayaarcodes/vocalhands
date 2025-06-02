import cv2
import os
import numpy as np
import json
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.applications.efficientnet import preprocess_input
import time
from gtts import gTTS
import pygame
import threading

pygame.init()
pygame.mixer.init()

with open('class_labels.json') as f:
    class_labels = json.load(f)

interpreter = tf.lite.Interpreter(model_path='sl_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

detector = HandDetector(maxHands=1)
cap = cv2.VideoCapture(0)

cv2.namedWindow("Hand Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hand Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

recognized_word = ""
current_letter = None
stable_count = 0
stability_threshold = 20
reset_time = 3
last_update_time = time.time()
blink_duration = 0.3
is_blinking = False

def play_audio(text):
    """Convert text to speech and play the audio without blocking."""
    tts = gTTS(text=text, lang='en')
    output_file_path = "output.mp3"
    tts.save(output_file_path)
    pygame.mixer.music.load(output_file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    os.remove(output_file_path)

try:
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        hands, img = detector.findHands(img, draw=False)

        if hands:
            for hand in hands:
                bbox = hand["bbox"]
                center = hand["center"]

                margin = 50
                x, y, w, h = bbox
                x_new = max(0, x - margin)
                y_new = max(0, y - margin)
                w_new = w + 2 * margin
                h_new = h + 2 * margin

                hand_region = img[y_new:y_new + h_new, x_new:x_new + w_new]
                hand_region_resized = cv2.resize(hand_region, (240, 240))
                hand_region_preprocessed = preprocess_input(hand_region_resized)
                sample_input = hand_region_preprocessed.reshape(1, 240, 240, 3).astype('float32')

                interpreter.set_tensor(input_details[0]['index'], sample_input)
                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]['index'])
                predicted_class_index = np.argmax(output_data, axis=1)[0]
                predicted_class_label = list(class_labels.keys())[list(class_labels.values()).index(predicted_class_index)]

                # Check for stability
                if predicted_class_label == current_letter:
                    stable_count += 1
                else:
                    stable_count = 0
                    current_letter = predicted_class_label

                # If stable for enough frames, process the recognized word
                if stable_count >= stability_threshold:
                    if current_letter == "space":
                        recognized_word += " "
                    elif current_letter == "del":
                        recognized_word = recognized_word[:-1]
                    else:
                        recognized_word += current_letter

                    last_update_time = time.time()
                    stable_count = 0
                    is_blinking = True
                    blink_start_time = time.time()

                blink_color = (0, 255, 0) if is_blinking else (255, 0, 0)
                blink_thickness = 4 if is_blinking else 2

                if is_blinking and time.time() - blink_start_time > blink_duration:
                    is_blinking = False

                cv2.rectangle(img, (x_new, y_new), (x_new + w_new, y_new + h_new), blink_color, blink_thickness)
                cv2.putText(img, predicted_class_label, (x_new, y_new - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        else:
            # Check if the reset time has been exceeded
            if time.time() - last_update_time >= reset_time and recognized_word:
                audio_thread = threading.Thread(target=play_audio, args=(recognized_word,))
                audio_thread.start()
                recognized_word = ""

        # Display the accumulated word
        cv2.putText(img, f"Word: {recognized_word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()