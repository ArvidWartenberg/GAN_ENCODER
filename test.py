import numpy as np

n = 100
ix = np.random.randint(0, 200, 100)
ix_sort = np.sort(ix)
all = np.arange(0, 200, 1)

for e in all'''
	for i in range(0,n_samples):
		tmp = np.copy(X_noise[i,:,:,0])
		x = np.random.randint(0,26,2)
		x1 = min(x)
		x2 = max(x)+1
		y = np.random.randint(0, 26, 2)
		y1 = min(y)
		y2 = max(y) + 1
		tmp[x1:x2, y1:y2] += np.random.uniform(-1,1,tmp[x1:x2,y1:y2].shape)
		tmp[x1:x2,y1:y2] = (tmp[x1:x2,y1:y2]-np.min(tmp[x1:x2,y1:y2]))/(np.max(tmp[x1:x2,y1:y2])-np.min(tmp[x1:x2,y1:y2]))
		X_noise[i,:,:,0] = tmp
	'''
	'''
	for i in range(0,gen_set_upd.shape[0]):
		tmp = np.copy(gen_set_upd[i, :, :, 0])
		x = np.random.randint(0, 26, 2)
		x1 = min(x)
		x2 = max(x) + 1
		y = np.random.randint(0, 26, 2)
		y1 = min(y)
		y2 = max(y) + 1
		tmp[x1:x2, y1:y2] += np.random.uniform(-1, 1, tmp[x1:x2, y1:y2].shape)
		tmp[x1:x2, y1:y2] = (tmp[x1:x2, y1:y2] - np.min(tmp[x1:x2, y1:y2])) / (
					np.max(tmp[x1:x2, y1:y2]) - np.min(tmp[x1:x2, y1:y2]))
		gen_set_upd[i, :, :, 0] = tmp
	'''

'''
def generate_training_samples(dataset, n_samples, g_model):
	ix_real = randint(0, dataset.shape[0], n_samples)
	ix_fake = randint(0, dataset.shape[0], n_samples)
	ix_noise = randint(0, dataset.shape[0], n_samples*2)

	X_real = np.copy(dataset[ix_real]) # real pix

	X_fake = np.copy(dataset[ix_fake]) # for generating fake pix
	X_fake = destroy_information(X_fake)
	X_fake = g_model.predict(X_fake)

	X_noise = np.copy(dataset[ix_noise]) # for training gan later (noise)

	y_real = ones((n_samples, 1))
	y_fake = ones((n_samples, 1)) * 0
	y_noise = ones((X_noise.shape[0], 1))
	return X_real, y_real, X_fake, y_fake, X_noise, y_noise
'''