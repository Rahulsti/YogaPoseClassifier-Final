# ---------------------------------------
# Yoga Pose Classifier (Transfer Learning)
# ---------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # To bypass decompression bomb warning for very large images
Image.LOAD_TRUNCATED_IMAGES = True  # Allows PIL to load truncated/incomplete files
# -----------------------------
# 1. Dataset Paths
# -----------------------------
train_dir = r"E:\Yoga\dataset\train"
val_dir   = r"E:\Yoga\dataset\val"

img_size = (224, 224)
batch_size = 32

# -----------------------------
# 2. Data Generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
)

# -----------------------------
# 3. Model (Transfer Learning)
# -----------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(train_data.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# -----------------------------
# 4. Callbacks
# -----------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint("yoga_pose_model.h5", save_best_only=True)

# -----------------------------
# 5. Training
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[early_stop, checkpoint]
)

# -----------------------------
# 6. Training Curves
# -----------------------------
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.show()

# -----------------------------
# 7. Prediction Function
# -----------------------------
from tensorflow.keras.preprocessing import image

def predict_pose(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    class_labels = list(train_data.class_indices.keys())
    return class_labels[class_idx]

# Example usage
print("Predicted Pose:", predict_pose(r"E:\Yoga\some_test_image.jpg"))
