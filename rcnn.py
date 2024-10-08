import tensorflow as tf
from tensorflow.keras import layers, models

# Load CIFAR-10 dataset
(X_train, y_train), (X_val, y_val) = tf.keras.datasets.cifar10.load_data()
X_train, X_val = X_train / 255.0, X_val / 255.0

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(64)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)

# Define RCNN model using transfer learning (ResNet50 as backbone)
def create_rcnn():
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    model = models.Sequential([
        layers.Resizing(224, 224),  # Resize CIFAR-10 images to 224x224
        base_model,
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 classes
    ])
    return model

# Compile and train the model
rcnn_model = create_rcnn()
rcnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
rcnn_model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Evaluate
rcnn_model.evaluate(val_dataset)
