#This is the code implementing the reverse diffusion model

import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras import layers

### Plotting functions

def plot_images(model, shape=(3,6), diffusion_steps=10):
    # plot random generated images for visual evaluation of generation quality
    num_rows,num_cols = shape
    generated_images = model.generate(
        num_images=num_rows * num_cols,
        diffusion_steps=diffusion_steps
    )

    plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
    for row in range(num_rows):
        for col in range(num_cols):
            index = row * num_cols + col
            plt.subplot(num_rows, num_cols, index + 1)
            plt.imshow(generated_images[index])
            #plt.imshow(x_train[index])
            plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()

def simple_plot(images):
    #plost a list of vectors of images
    num_rows = len(images)
    num_cols = images[0].shape[0]
    plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
    for row in range(num_rows):
        for col in range(num_cols):
            index = row * num_cols + col
            plt.subplot(num_rows, num_cols, index + 1)
            plt.imshow(images[row][col])
            #plt.imshow(x_train[index])
            plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()

### Network's Blocks

def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings

    
def ResidualBlock(depth):
    def apply(x):
        input_depth = x.shape[3]
        if input_depth == depth:
            residual = x
        else:
            residual = layers.Conv2D(depth, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            depth, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(depth, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(depth, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(depth)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(depth, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(depth)(x)
        return x

    return apply


def get_denoising_network(image_size, depths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(depths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for depth in depths[:-1]:
        x = DownBlock(depth, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(depths[-1])(x)

    for depth in reversed(depths[:-1]):
        x = UpBlock(depth, block_depth)([x, skips])

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="denoising")


def get_Unet(image_size, depths, block_depth):
    input_images = keras.Input(shape=(image_size, image_size, 3))

    x = layers.Conv2D(depths[0], kernel_size=1)(input_images)

    skips = []
    for depth in depths[:-1]:
        x = DownBlock(depth, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(depths[-1])(x)

    for depth in reversed(depths[:-1]): 
        x = UpBlock(depth, block_depth)([x, skips])

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model(input_images, x, name="unet")

### Dataset

#(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
x_train = np.load('/home/andrea/CELEBA/data/celeba/train.npy')
x_test = np.load('/home/andrea/CELEBA/data/celeba/test.npy')
x_train = (x_train/255.).astype(np.float32)
x_test = (x_test/255.).astype(np.float32)

### Network hyperparameters

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
image_size = 64
embedding_dims = 64
embedding_max_frequency = 1000.0
#depths = [32, 64, 96, 128]
#depths = [32, 64, 128, 256]
depths = [48, 96, 192, 384]
block_depth = 2

### The diffusion model

class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth, batch_size):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_denoising_network(image_size, widths, block_depth)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")

    def load_weights(self, filename):
        self.network.load_weights(filename)

    def save_weights(self, filename):
        self.network.save_weights(filename)

    @property
    def metrics(self):
        return [self.noise_loss_tracker, 
                self.image_loss_tracker 
                ]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        network = self.network
        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        if num_images is None:
            num_images = 10
        step_size = 1.0 / diffusion_steps

        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, image_size, image_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=batch_size, diffusion_steps=kid_diffusion_steps
        )

        return {m.name: m.result() for m in self.metrics}  

### Training hyperparameters
batch_size = 16
learning_rate = 0.0001
weight_decay = 1e-4

num_epochs = 5

### Create and compile the model
model = DiffusionModel(image_size, depths, block_depth, batch_size)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.mean_absolute_error,
)

#model.network.summary()

# calculate mean and variance of training dataset for normalization
model.normalizer.adapt(x_test)

### Run training and plot generated images periodically

weights_filename = "diffusion_celeba_github.hdf5"
model.load_weights("weights/"+weights_filename)

if num_epochs > 0:
    model.fit(
        #x_train[0:64000],
        x_train[64000:128000],
        #x_train[128000:162752],
        #x_test[0:19952],
        epochs=num_epochs,
        batch_size=batch_size
    )
    model.save_weights("weights/"+weights_filename)

# current_loss = 0.1082
plot_images(model,diffusion_steps=10)


### Gradient descent
### The trainable part of the model is the initial noise, added as weight
### as part of the build method
### The initial noise can be randomly initialized (default),
### or it can be passed as input

class DescentModel(keras.Model):
    def __init__(self, reference_model, n, starting_noise = None):
        super().__init__()
        self.reference_model = reference_model
        self.n = n
        self.starting_noise = starting_noise
        self.reference_model.trainable = False #the model must not be trainable
    
    def build(self, input_shape):
        if self.starting_noise is None:
            #Warning the default keras stddev is 0.05 
            initializer = tf.keras.initializers.RandomNormal(mean=0.,stddev=1.)
        else:
            initializer = tf.keras.initializers.Constant(self.starting_noise)
        self._initial_noise = self.add_weight(name='initial_noise', shape=(self.n,)+input_shape[1:], initializer=initializer, trainable=True)
  
    def call(self, inputs):
        #we add a small quadratic penalty
        self.add_loss(.005 * tf.reduce_mean(self._initial_noise**2))
        generated_images = self.reference_model.reverse_diffusion(self._initial_noise, 10)
        generated_images = self.reference_model.normalizer.mean + generated_images * self.reference_model.normalizer.variance**0.5
        return generated_images
    
    def compute_output_shape(self, input_shape):
        return (self.n,) + input_shape[1:]
 
def loss_function(ground_truth, predicted):
    return tf.reduce_mean(tf.math.abs(ground_truth - predicted))

def diffusion_descent(ground_truth,n,starting_noise=None, epochs=1500):
    ground_truth = tf.expand_dims(ground_truth, 0)
    descent_model = DescentModel(model,n,starting_noise=starting_noise)
    descent_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-2),
        loss=loss_function
    )
    descent_model.fit(
        x=ground_truth,
        y=ground_truth,
        epochs=epochs
    )

    output = descent_model.predict(
        ground_truth
    )

    return output, descent_model._initial_noise  

def generate_seeds(source_img_no,num_seeds=16,save=True,epochs=1500):
    #generate seeds for source_img and save them
    reference_image = x_test[img_no]
    #plt.imshow(reference_image)
    #plt.show()
    outputs, seeds = diffusion_descent(reference_image,num_seeds,epochs=epochs)
    filename = 'seeds_for_celeba_test_'+str(img_no)+'.npy'
    if save:
        with open(filename, 'wb') as f:
            np.save(f,seeds)

### Embedding Network

def train_embedding_network(embedding_model,generative_model,trainset,epochs,batch_size):
    generative_model.trainable = False
    def visible_loss(y_true,y_pred):
        #y_pred is the seed returned by the model
        generated = generative_model.reverse_diffusion(y_pred, 10)
        generated = generative_model.denormalize(generated)
        err = tf.reduce_mean(tf.math.abs(y_true - generated))
        return err
    optimizer=keras.optimizers.Adam(learning_rate=.0001)
    embedding_model.compile(optimizer=optimizer,loss=visible_loss)
    embedding_model.fit(trainset,trainset,epochs=epochs,batch_size=batch_size)

#import embedding_models
embedding_model = get_Unet(64,depths,block_depth)
weights_name = 'weights/networkUnet_full.hdf5'
embedding_model.load_weights(weights_name)
trainset = x_train[100000:164000]
train_embedding_network(embedding_model,model,trainset,5,16)
embedding_model.save_weights(weights_name)
    
    
    

