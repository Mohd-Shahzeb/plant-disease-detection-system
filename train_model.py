import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import json

# Dataset path
data_dir = "dataset"

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generator
train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save class names
class_names = list(train_data.class_indices.keys())
print("Classes:", class_names)

# Load pretrained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🔥 TRAIN (FAST SETTINGS)
model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    steps_per_epoch=50,
    validation_steps=20
)

# Save model
model.save("model/plant_model.h5")

# Save class labels
with open("model/classes.json", "w") as f:
    json.dump(class_names, f)

print("Training Complete!")