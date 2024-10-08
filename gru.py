import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import sequence

# Load IMDB dataset
(X_train, y_train), (X_val, y_val) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Pad sequences to ensure they are the same length
max_len = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_val = sequence.pad_sequences(X_val, maxlen=max_len)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(64)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)

# Define the GRU model
def create_gru():
    model = models.Sequential([
        layers.Embedding(input_dim=10000, output_dim=128, input_length=max_len),
        layers.GRU(128, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    return model

# Compile and train the GRU model
gru_model = create_gru()
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gru_model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# Evaluate
gru_model.evaluate(val_dataset)
