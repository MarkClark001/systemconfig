import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import train_test_split
import os

# Path to your image directory (Make sure to structure it in subdirectories per class for classification)
image_directory = '/path_to_image_directory'

# Load dataset from the directory and split into train/test using image_dataset_from_directory
dataset = image_dataset_from_directory(
    image_directory,
    image_size=(224, 224),  # Resize all images to a consistent size (for example, 224x224)
    batch_size=32,          # Batch size for loading
    label_mode='int',       # Integer labels for classification, use 'categorical' for one-hot encoding
    shuffle=True            # Shuffle images while loading
)

# Splitting the dataset into 80% train and 20% test
train_size = 0.8
train_dataset = dataset.take(int(len(dataset) * train_size))
test_dataset = dataset.skip(int(len(dataset) * train_size))

# Further split training set into train and validation (e.g., 80/20 split)
train_dataset = train_dataset.take(int(len(train_dataset) * train_size))
val_dataset = train_dataset.skip(int(len(train_dataset) * train_size))

# Preprocess the images (rescaling)
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Data augmentation (optional, for train set only)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))

# Caching and prefetching to improve loading speed
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Ready dataloaders for model training
print(train_dataset, val_dataset, test_dataset)
