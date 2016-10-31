import numpy as np
import cv2
import matplotlib.pyplot as plt
import cPickle, gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


def read_images_from_mnist(file_path):
	f = gzip.open(file_path, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	for i in xrange(len(train_set[0])):
		img = np.array(train_set[0][i]).reshape(28,28)
		img = normalize_img(img)
		img = whitenning_img(img)
		train_set[0][i] = img.reshape(784)

	for i in xrange(len(valid_set[0])):
		img = np.array(valid_set[0][i]).reshape(28,28)
		img = normalize_img(img)
		img = whitenning_img(img)
		valid_set[0][i] = img.reshape(784)

	for i in xrange(len(test_set[0])):
		img = np.array(test_set[0][i]).reshape(28,28)
		img = normalize_img(img)
		img = whitenning_img(img)
		test_set[0][i] = img.reshape(784)

	return train_set, valid_set, test_set

def normalize_img(img):
	return cv2.normalize(img, img, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

def whitenning_img(img):
	return cv2.Sobel(img,cv2.CV_32F,1,1,ksize=5)

def sampling_image(img, window_size=(5,5)):
	x = np.random.randint(0, img.shape[0]-window_size[0])
	y = np.random.randint(0, img.shape[1]-window_size[1])
	return img[x : x + window_size[0], y: y + window_size[1]].flatten()

def generate_samples(data_set, num_samples, window_size=(5,5)):
	samples = []
	for i in range(num_samples):
		random_img_posi = np.random.randint(0, len(data_set))
		img = data_set[random_img_posi]
		sample = sampling_image(img.reshape(28,28), window_size=window_size )
		samples.append(sample)
	return np.array(samples)

def generate_cluster_centroids(data_set):
	X = np.array(data_set[0])
	y = data_set[1]

	window_size = 5
	num_cluster = 16
	sqrt_num_cluster = int(np.sqrt(num_cluster))

	print("GENERATING SAMPLES")
	samples = generate_samples(X, 10000, window_size=(window_size,window_size))

	print("FITTING K-MEANS")
	kmeans_model =  MiniBatchKMeans(n_clusters=num_cluster, batch_size=100, n_init=10, max_no_improvement=10, verbose=False)
	kmeans_model.fit(samples)

	
	labels = kmeans_model.labels_
	centroids = kmeans_model.cluster_centers_


	# print "NUM CENTROIDS: ", len(centroids)
	# fig, subs = plt.subplots(sqrt_num_cluster, sqrt_num_cluster)
	# for i in xrange(sqrt_num_cluster):
	# 	for j in xrange(sqrt_num_cluster):
	# 		subs[i][j].imshow(centroids[i*j].reshape(window_size,window_size),  cmap='Greys_r')
	# 		subs[i][j].axis('off')
	# plt.show()


	for x in X:
		for center in centroids:
			res = cv2.filter2D(x.reshape(28,28), -1, center.reshape(window_size, window_size))
			print res
			plt.imshow(res, cmap='Greys_r')
			plt.show()
			# res.flatten()
	


def run():
	train_set, valid_set, test_set = read_images_from_mnist("../mnist/mnist.pkl.gz")
	generate_cluster_centroids(train_set)




if __name__ == '__main__':
	run()