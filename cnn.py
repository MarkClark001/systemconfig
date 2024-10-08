import tensorflow as tf
from tensorflow.keras import layers, models

# Load CIFAR-10 dataset
(X_train, y_train), (X_val, y_val) = tf.keras.datasets.cifar10.load_data()
X_train, X_val = X_train / 255.0, X_val / 255.0

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(64)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)

# Define the CNN model
def create_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Compile and train the model
cnn_model = create_cnn()
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Evaluate
cnn_model.evaluate(val_dataset)
