import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Directories and parameters
TRAIN_DIR = '/path/to/sl_dataset'
SAVE_DIR = '/path/to/vocalhands/src'
TARGET_SIZE = (260, 260)
N_CLASSES = 28
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

data_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    brightness_range=[0.7, 1.4],
    validation_split=VALIDATION_SPLIT
)

train_generator = data_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = data_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

class_indices = train_generator.class_indices
class_names = {v: k for k, v in class_indices.items()}
with open(os.path.join(SAVE_DIR, 'class_labels.json'), 'w') as f:
    json.dump(class_indices, f)

def visualize_data(generator):
    class_images = {class_name: None for class_name in class_names.values()}

    for img, labels in generator:
        for i, label in enumerate(labels):
            class_label = np.argmax(label)
            class_name = class_names[class_label]

            if class_images[class_name] is None:
                class_images[class_name] = img[i]

        if all(img is not None for img in class_images.values()):
            break

    plt.figure(figsize=(20, 20))
    for i, (class_name, img) in enumerate(class_images.items()):
        plt.subplot(8, 4, i + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_data(train_generator)

def build_model():
    input_tensor = Input(shape=(260,260,3))
    base_model = EfficientNetB2(include_top=False, weights='imagenet', input_tensor=input_tensor)

    base_model.trainable = False

    for layer in base_model.layers[-150:]:
      if not isinstance(layer, BatchNormalization):
        layer.trainable = True

    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(1024, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)

    x = Dense(512, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)

    x = Dense(256, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)

    output = Dense(N_CLASSES, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer=Adam(learning_rate=7e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = build_model()
model.summary()

callbacks = [
    ModelCheckpoint(os.path.join(SAVE_DIR, 'best_model.keras'), save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-10, verbose=1)
]

history = model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=callbacks)

model.save(os.path.join(SAVE_DIR, 'sl_model.keras'))

# Plot training and validation history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_training_history(history)