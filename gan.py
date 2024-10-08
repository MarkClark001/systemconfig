import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Create Generator model
def create_generator():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dense(32*32*3, activation='tanh'),
        layers.Reshape((32, 32, 3))
    ])
    return model

# Create Discriminator model
def create_discriminator():
    model = models.Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(128),
        layers.LeakyReLU(),
        layers.Dense(64),
        layers.LeakyReLU(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# GAN model
generator = create_generator()
discriminator = create_discriminator()

# Compile models
generator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# GAN setup
gan_input = layers.Input(shape=(100,))
generated_image = generator(gan_input)
discriminator.trainable = False
gan_output = discriminator(generated_image)
gan = models.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training loop (simplified)
batch_size = 64
epochs = 10000
latent_dim = 100

for epoch in range(epochs):
    # Train discriminator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_images = generator.predict(noise)
    real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
    
    labels_real = np.ones((batch_size, 1))
    labels_fake = np.zeros((batch_size, 1))

    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(real_images, labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)

    # Train generator via GAN
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gan_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, GAN loss: {gan_loss}, D Loss: {d_loss_real + d_loss_fake}')
