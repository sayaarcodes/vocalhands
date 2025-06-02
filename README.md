# VocalHands

**VocalHands** is a real-time American Sign Language (ASL) recognition system designed to break down the communication barrier between people who rely on sign language and those who do not understand it.

---

## Vision

I created VocalHands to address a critical communication gap: individuals who are deaf or hard of hearing use ASL to communicate but often face barriers when others do not understand sign language. VocalHands captures sign language text and voices it for effective communication. This isn’t just a “nice-to-have” feature; it can fundamentally improve how people connect, share ideas, and feel included. I built this for my science fair in high school, where it won 1st prize, and my hope is that VocalHands will continue evolving, eventually running on mobile devices. By making ASL fully audible and accessible to non-sign language users, VocalHands aims to empower sign language users and eliminate the isolation that often comes from communication hurdles.

---

## Repository Structure

```
vocalhands/
├── LICENSE
├── README.md
├── requirements.txt
├── train/
│   └── train.py             # Training script and related model definitions
├── src/
│   ├── main.py              # Real-time ASL→speech program
│   ├── sl_model.tflite      # TensorFlow Lite model for inference
│   └── class_labels.json    # Mapping of ASL labels to class indices
```

---

## Features

- **Real-time hand detection**  
  Uses webcam input and OpenCV to locate a single hand in each frame.
- **ASL letter/word recognition (TF-Lite)**  
  A lightweight TensorFlow Lite model classifies hand shapes into ASL letters. Over time, these letters form words.
- **Speech synthesis**  
  Once a word remains stable for a few frames, VocalHands converts it to speech using gTTS and pygame.
- **Visual feedback**  
  A colored rectangle flashes around the detected hand to indicate when a character is recognized (“blinks” in green), and the current letter appears above the box. The accumulated word displays at the top-left of the screen.

---

## Installation

1. **Clone this repository**
   ```
   git clone https://github.com/sayaarcodes/vocalhands.git
   cd vocalhands
   ```

2. **Create a virtual environment (recommended)**
   ```
   python -m venv venv
   # Activate the environment:
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

   > **Note:** If you only plan to run inference, you can streamline dependencies by installing:
   > ```
   > pip install tensorflow-lite opencv-python cvzone gTTS pygame numpy
   > ```
   > For training, you’ll need the full TensorFlow package (e.g., `tensorflow`), plus any additional libraries (`matplotlib`, etc.).

---

## Training

The `train/train.py` script defines and trains an ASL classifier based on EfficientNetB2. Follow these steps to train your own model:

1. **Prepare your dataset**  
   - Organize your ASL images in a directory with the following structure:
     ```
     sl_dataset/
     ├── A/
     │   ├── img1.jpg
     │   ├── ...
     ├── B/
     │   ├── ...
     └── ...
     ```
   - Each subfolder (e.g., `A`, `B`, ..., `space`, `del`) should contain images for that class.

2. **Update paths in `train/train.py`**  
   In `train/train.py`, adjust:
   ```python
   TRAIN_DIR = '/path/to/sl_dataset'
   SAVE_DIR = '/path/to/vocalhands/src'  # Where trained models and labels will be saved
   ```
   or use relative paths:
   ```python
   TRAIN_DIR = '../data/sl_dataset'      # if you place data under a `data/` folder at repo root
   SAVE_DIR = '../src'
   ```

3. **Run the training script**
   ```
   python train/train.py
   ```
   - The script will:
     - Load images via `ImageDataGenerator`, applying preprocessing and augmentation.
     - Save `class_labels.json` in `src/`.
     - Build an EfficientNetB2-based model (with custom dense layers and dropout).
     - Train for 20 epochs (adjustable in the script).
     - Save the best model weights as `best_model.keras` and the final model as `sl_model.keras` in `src/`.
     - Plot training/validation accuracy and loss at the end.

4. **Convert to TFLite**  
   After training, convert your saved Keras model to TensorFlow Lite for inference:
   ```python
   import tensorflow as tf

   keras_model = tf.keras.models.load_model('src/sl_model.keras')
   converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
   tflite_model = converter.convert()
   with open('src/sl_model.tflite', 'wb') as f:
       f.write(tflite_model)
   ```
   This produces `sl_model.tflite` in the `src/` directory.

5. **Verify `class_labels.json`**  
   Ensure that `train/train.py` saved `class_labels.json` in `src/`. This file maps ASL letters (and special tokens `space`/`del`) to numerical class indices for inference.

---

## Usage (Inference)

1. **Run the main script**
   ```
   cd src
   python main.py
   ```

2. **Allow webcam access**  
   The program will open a fullscreen window named **Hand Detection**. Position your hand in front of the camera.

3. **Sign letters/words**  
   - Hold up ASL letters (A–Z).  
   - To form a space, hold the **space** sign.  
   - To delete the last character, hold the **del** sign.  
   - Once a letter is held steadily (≈20 frames), it appends to the current word.  
   - After you pause (≈3 seconds of no new signs), the full word is converted to speech automatically.

4. **Quit**  
   Press **q** in the window to stop the program. Any ongoing audio playback will finish before closing.

---

## Limitations

- **Desktop only**  
  VocalHands runs on PC or laptop environments (Windows, macOS, Linux). Mobile support (iOS/Android) is not available right now.
- **Single-hand tracking**  
  Only one hand can appear in the frame. If both hands or extra objects are detected, recognition will fail.
- **Lighting & background sensitivity**  
  Bright, even lighting helps the hand detector work reliably. Dark or cluttered backgrounds may cause misclassification.
- **Vocabulary scope**  
  The current model recognizes basic ASL letters (A–Z), **space**, and **delete**. It does not handle full ASL grammar or finger‑spelling beyond letters/words.
- **Latency & performance**  
  On lower‑end machines, the frame rate might drop below real time (≈10–15 FPS). For best results, use a computer with a decent CPU/GPU.
- **Pronunciation quirks**  
  Because speech is generated with Google Text‑to‑Speech (gTTS), some words might sound robotic or pronounce certain letter combinations awkwardly.

---

## Future Work

- **Mobile support**  
  Adapt the hand detector and inference engine for smartphones (TensorFlow Lite on Android/iOS).
- **Expanded ASL dictionary**  
  Train a more complex model to recognize common ASL words/phrases instead of letter‑by‑letter spelling.
- **Gesture smoothing & auto‑correction**  
  Improve stability thresholds and introduce a simple spell‑check to fix minor recognition errors.
- **User interface**  
  Build a lightweight GUI (perhaps via Flask or Electron) so non‑technical users can install and run VocalHands with a single click.
- **Customizable voice settings**  
  Allow users to choose different TTS voices, speech rates, or even local offline TTS engines.

---

## Acknowledgments

- **cvzone.HandTrackingModule** by @cvzone – for making hand detection straightforward.  
- **TensorFlow Lite** – for enabling a lightweight ASL letter classifier.  
- **gTTS & pygame** – for quick and easy speech synthesis/playback.  
- **Inspired by** families and peers who deserve better communication tools.

---

## License

This project is released under the GNU GENERAL PUBLIC License. See [LICENSE](LICENSE) for details.
