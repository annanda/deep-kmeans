import cv2
import cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt


# description: read image from mnist folder
# input:
#		file_path: path for mnist cpickled file
# output:
#		train_set (50k examples): [ [list of images], [list of labels related with each image]  ] 
#		valid_set (10k examples): [ [list of images], [list of labels related with each image]  ] 
#		test_set (10k examples): [ [list of images], [list of labels related with each image]  ] 
#
# observation:	image are normalized and whitenning following paper's 
# 				instructions (http://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf)
#
def read_images_from_mnist(file_path, normalize=True, whitenning=True):
	
	f = gzip.open(file_path, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	for i in xrange(len(train_set[0])):
		img = np.array(train_set[0][i]).reshape(28,28)

		if normalize:
			img = normalize_img(img)

		if whitenning:			
			img = whitenning_img(img)

		train_set[0][i] = img.reshape(784)

	for i in xrange(len(valid_set[0])):
		img = np.array(valid_set[0][i]).reshape(28,28)

		if normalize:
			img = normalize_img(img)

		if whitenning:
			img = whitenning_img(img)

		valid_set[0][i] = img.reshape(784)

	for i in xrange(len(test_set[0])):
		img = np.array(test_set[0][i]).reshape(28,28)

		if normalize:
			img = normalize_img(img)

		if whitenning:
			img = whitenning_img(img)

		test_set[0][i] = img.reshape(784)

	return train_set, valid_set, test_set


# description: normalize values inside an image matrix 
# input:
#		img: numpy array
# output:
#		img: numpy array 
def normalize_img(img):
	return cv2.normalize(img, img, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

# description: remove content (filter) image using a sobel filter
# input:
#		img: numpy array
# output:
#		img: numpy array 
def whitenning_img(img):
	return cv2.Sobel(img,cv2.CV_32F,1,1,ksize=5)

# description: get a sample slice from a image
# input:
#		img: numpy array
#		window_size: a tuple (width, height) with the sample size
# output:
#		img: numpy array 
def sampling_image(img, window_size=(5,5)):
	x = np.random.randint(0, img.shape[0]-window_size[0])
	y = np.random.randint(0, img.shape[1]-window_size[1])
	return img[x : x + window_size[0], y: y + window_size[1]].flatten()

# description: from a set of images, generate many samples (slices)
# input:
#		dataset: numpy array with multiple images (each image is a numpy array)
#		num_samples: number of samples that must be generated
#		window_size: a tuple (width, height) with the sample size
# output:
#		img: numpy array with multiple image samples
def generate_samples(data_set, num_samples, window_size=(5,5)):
	samples = []
	for i in range(num_samples):
		random_img_posi = np.random.randint(0, len(data_set))
		img = data_set[random_img_posi]
		sample = sampling_image(img.reshape(28,28), window_size=window_size )
		samples.append(sample)
	return np.array(samples)

# description: draw multiple images (usually k-means centroids) in a grid
# input:
#		images: numpy array with multiple images (each image is a numpy array)
#		num_lines: number of lines in grid
#		num_columns: number of columns in grid
def draw_multiple_images(images, num_lines, num_columns):

	img_is_flatten = True if len(images[0].shape) == 1 else False

	fig, subs = plt.subplots(num_lines, num_columns)
	for i in xrange(num_lines):
		for j in xrange(num_columns):
			img = images[i*j]

			if img_is_flatten:

				width = np.sqrt(img.shape[0])
				height = np.sqrt(img.shape[0])
				subs[i][j].imshow(img.reshape(width, height),  cmap='Greys_r')

			else:
				subs[i][j].imshow(img,  cmap='Greys_r')

			subs[i][j].axis('off')

	plt.show()

# description: draw an image
# input:
#		image: numpy array 
def draw_img(img):
	plt.imshow(img, cmap='Greys_r')
	plt.show()

# description: apply a filter over a imge
# input:
#		img: numpy array representing original image
#		filter: numpy array representing the filter that will be apply
# output:
#		img: numpy array with a filtered image
def convolve_image(img, filter):
	return cv2.filter2D(img, -1, filter)