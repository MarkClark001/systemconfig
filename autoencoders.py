import tensorflow as tf
from tensorflow.keras import layers, models

# Load CIFAR-10 dataset
(X_train, _), (X_val, _) = tf.keras.datasets.cifar10.load_data()
X_train, X_val = X_train / 255.0, X_val / 255.0

train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(10000).batch(64)
val_dataset = tf.data.Dataset.from_tensor_slices(X_val).batch(64)

# Define the Autoencoder model
def create_autoencoder():
    encoder = models.Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu')
    ])
    
    decoder = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(64,)),
        layers.Dense(32*32*3, activation='sigmoid'),
        layers.Reshape((32, 32, 3))
    ])
    
    autoencoder = models.Sequential([encoder, decoder])
    return autoencoder

# Compile and train the autoencoder
autoencoder_model = create_autoencoder()
autoencoder_model.compile(optimizer='adam', loss='mse')
autoencoder_model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Evaluate
autoencoder_model.evaluate(val_dataset)
