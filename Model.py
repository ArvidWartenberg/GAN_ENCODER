'''import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
'''
# example of training a gan on mnist
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
import numpy as np
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Reshape, Input
from keras.layers import Conv2D, Convolution2D, UpSampling2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.models import Model
from matplotlib import pyplot
from keras import backend as K
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt


# define the standalone discriminator model
def convolutional_autoencoder(input_img = Input(shape=(28, 28, 1))):
	'''x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)  # nb_filter, nb_row, nb_col
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='valid')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Convolution2D(1, 5, 5, activation='sigmoid', border_mode='same')(x)
	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	'''
	x = Conv2D(32, (3, 3), activation='relu', padding ='same')(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(32, (3, 3), activation='relu', padding ='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(32, (3, 3), activation='relu', padding ='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(32, (3, 3), activation='relu', padding ='same')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding ='same')(x)
	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adam', loss ='binary_crossentropy')
	print("shape of encoded", K.int_shape(encoded))
	print("shape of decoded", K.int_shape(decoded))
	return autoencoder

def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load and prepare mnist training images
def load_real_samples():
	# load mnist dataset
	(trainX, _), (_, _) = load_data()
	# expand to 3d, e.g. add channels dimension
	X = expand_dims(trainX, axis=-1)
	# convert from unsigned ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [0,1]
	X = X / 255.0
	return X

def destroy_information(dataset):
	for i in range(0,dataset.shape[0]):
		tmp = np.copy(dataset[i,:,:,0])
		x = np.random.randint(0,26,2)
		x1 = min(x)
		x2 = max(x)+1
		y = np.random.randint(0, 26, 2)
		y1 = min(y)
		y2 = max(y) + 1
		tmp[x1:x2, y1:y2] += np.random.uniform(0,1,tmp[x1:x2,y1:y2].shape)
		tmp[x1:x2,y1:y2] = (tmp[x1:x2,y1:y2]-np.min(tmp[x1:x2,y1:y2]))/(np.max(tmp[x1:x2,y1:y2])-np.min(tmp[x1:x2,y1:y2]))
		dataset[i, :, :, 0] = tmp
	return dataset

def generate_real_samples(dataset, n_samples):
	ix_real = randint(0, dataset.shape[0], n_samples)
	X_real = np.copy(dataset[ix_real])  # real pix
	y_real = ones((n_samples, 1))
	return X_real, y_real

def generate_fake_samples(dataset, n_samples, g_model, noise):
	ix_fake = randint(0, dataset.shape[0], n_samples)
	X_fake = np.copy(dataset[ix_fake])  # for generating fake pix
	if noise:
		X_fake = destroy_information(X_fake)
	X_fake = g_model.predict(X_fake)
	y_fake = ones((n_samples, 1)) * 0
	return X_fake, y_fake

def generate_destroyed_samples(dataset, n_samples, noise): # want 2x normal nsamples
	ix_noise = randint(0, dataset.shape[0], n_samples)
	X_noise = dataset[ix_noise]
	if noise:
		X_noise = np.copy(dataset[ix_noise]) # for training gan later (noise)
	y_noise = ones((X_noise.shape[0], 1))
	return X_noise, y_noise

def save_plot(examples, epoch, n=10):
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    plt.savefig(filename)
    plt.close()

def summarize_performance(epoch, g_model, d_model, dataset, noise, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(dataset, n_samples, g_model, noise)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(x_fake, epoch)
    save_plot(X_real, epoch+1)
    # save the generator model tile file
    filename = 'generator_model_srs_%03d.h5' % (epoch + 1)
    g_model.save(filename)

def train(g_model, d_model, gan_model, dataset, n_epochs=100, n_batch=256):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	noise = False
	for i in range(n_epochs):
		if i > 20:
			noise = True
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# Gen training data
			#X_real, y_real, X_fake, y_fake, X_noise, y_noise = generate_training_samples(dataset, half_batch, g_model)
			X_real, y_real = generate_real_samples(dataset, half_batch) # real samples
			X_fake, y_fake = generate_fake_samples(dataset, half_batch, g_model, noise) # fake
			X_noise, y_noise = generate_destroyed_samples(dataset, half_batch*2, noise) # for gen train

			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)

			#input some noisy imgs to generator
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_noise, y_noise)
			# summarize loss on this batch
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
		if (i + 1) % 2 == 0:
			summarize_performance(i, g_model, d_model, dataset, noise)

X = load_real_samples()
d_model = define_discriminator()
# create the generator
g_model = convolutional_autoencoder()
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
dataset = dataset[:,:,:,:]
# train model
train(g_model, d_model, gan_model, dataset,n_epochs=100)

'''
import matplotlib.pyplot as plt
for i in range(0,16):
	plt.subplot(4,4,i+1)
	plt.axis('off')
	plt.imshow(X_real[i,:,:,0])

plt.figure()
for i in range(0,16):
	plt.subplot(4,4,i+1)
	plt.axis('off')
	plt.imshow(X_fake[i,:,:,0])
plt.show()
print()
'''