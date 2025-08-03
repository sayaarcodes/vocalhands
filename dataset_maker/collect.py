import cv2
from cvzone.HandTrackingModule import HandDetector
import os
import numpy as np
import random
import time

while True:
    alphabet = input("Enter the alphabet to collect data for (e.g., A, B, C): ").upper()
    num_images = int(input("Enter the number of images to collect: "))

    output_folder = f'dataset/{alphabet}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    detector = HandDetector(maxHands=1)
    cap = cv2.VideoCapture(0)
    image_count = 0

    cv2.namedWindow("Hand Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Hand Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    prev_bbox = None
    stabilization_counter = 0
    required_stable_frames = 60
    stability_threshold = 10
    stabilized = False

    while image_count < num_images:
        success, img = cap.read()

        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, draw=False)

        if hands:
            hand = hands[0]  # first detected hand
            bbox = hand["bbox"]
            x, y, w, h = bbox

            # margin
            margin = 50
            x_new = max(0, x - margin)
            y_new = max(0, y - margin)
            w_new = w + 2 * margin
            h_new = h + 2 * margin

            # initial stabilization
            if not stabilized:
                if prev_bbox:
                    x_diff = abs(x - prev_bbox[0])
                    y_diff = abs(y - prev_bbox[1])
                    w_diff = abs(w - prev_bbox[2])
                    h_diff = abs(h - prev_bbox[3])

                    if x_diff < stability_threshold and y_diff < stability_threshold and w_diff < stability_threshold and h_diff < stability_threshold:
                        stabilization_counter += 1
                    else:
                        stabilization_counter = 0  # reset if movement is detected

                prev_bbox = (x, y, w, h)

                if stabilization_counter >= required_stable_frames:
                    print("Hand stabilized, starting to save images...")
                    stabilized = True

            else:
                hand_image = img[y_new:y_new + h_new, x_new:x_new + w_new]

                aspect_ratio = w_new / h_new
                new_width = 200
                new_height = int(new_width / aspect_ratio) if aspect_ratio > 1 else 200
                if new_height > 200:
                    new_height = 200
                    new_width = int(new_height * aspect_ratio)

                hand_image_resized = cv2.resize(hand_image, (new_width, new_height))

                final_image = np.zeros((200, 200, 3), dtype=np.uint8)

                x_offset = (200 - new_width) // 2
                y_offset = (200 - new_height) // 2
                final_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = hand_image_resized

                cv2.imwrite(os.path.join(output_folder, f'hand_{round(random.random() * time.time())}.jpg'), final_image)
                image_count += 1
                print(f"Saved image {image_count}/{num_images} for alphabet {alphabet}")

            cv2.rectangle(img, (x_new, y_new), (x_new + w_new, y_new + h_new), (255, 0, 0), 2)

        cv2.imshow("Hand Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Image collection complete.")
