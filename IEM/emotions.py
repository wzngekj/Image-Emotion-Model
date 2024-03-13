import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.applications import vgg16
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# initialising constants
EPOCHS = 20
IMG_WIDTH = 48
IMG_HEIGHT = 48
NUM_CLS = 7
BATCH_SIZE = 32

# loading and processing of data
def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                image = image.astype('float32') / 255.0  # Normalize to [0, 1]
                images.append(image)
                labels.append(label)

    label_map = {label: i for i, label in enumerate(np.unique(labels))}
    labels = [label_map[label] for label in labels]

    return np.array(images), np.array(labels)

# validity of command line
if len(sys.argv) not in [2, 3]:
    sys.exit("Usage: python traffic.py data_directory [model.h5]")

# loading of data into images and label
data_dir = sys.argv[1]
images, labels = load_data(data_dir)

# early stop
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# model checkpoint
checkpoint_path = "best_model.keras"

model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy', 
    mode='max',            
    save_best_only=True,    
    verbose=1      
)

# preparation and splitting of data
labels = tf.keras.utils.to_categorical(labels)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.4)

# loading of pre-trained vgg16 model
vgg = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

for layer in vgg.layers:
    layer.trainable = False

# model structure
model = Sequential([
    vgg,
    Flatten(),
    Dense(1024, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLS, activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=[early_stopping, model_checkpoint])

model.evaluate(x_test,  y_test, verbose=2)

model.summary()

# plotting training and validation loss values
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# plotting training and validation accuracy values
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()