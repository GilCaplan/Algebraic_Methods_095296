import pickle
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler


def unpickle(file):
    with open(file, 'rb') as fo:
        images = pickle.load(fo, encoding='bytes')
    return images


def process(s):
    img = Image.fromarray(s.reshape((3, 32, 32)).transpose((1, 2, 0)).astype('uint8')).convert("L")
    return np.array(img)


def process_batch(path):
    pics = unpickle(path)
    # Step 1: convert all images to gray scale
    images = pics[b'data']
    labels = pics[b'fine_labels']
    reshaped_images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    grayscale_images = []

    for image in reshaped_images:
        img = Image.fromarray(image.astype('uint8')).convert("L")
        grayscale_images.append(np.array(img).reshape(-1))

    return np.array(grayscale_images).reshape(-1, 1024), labels


grey_images_train, train_labels = process_batch(path := r'.\cifar-100-python\train')
grey_images_test, test_labels = process_batch(r'.\cifar-100-python\test')

# check dimensions of the target labels
print(f'Dimension of the target train labels: {len(train_labels)}')
print(f'Dimension of the target test labels: {len(test_labels)}')
print(f'Dimension of the original images: {grey_images_train.shape}')


def run_PCA_KNN_ondata(s_vals, k_vals):
    for s, k in zip(s_vals, k_vals):
        pca = PCA(n_components=s)

        # project the images onto the first s principal components
        pca.fit(grey_images_train)
        pca.fit(grey_images_test)
        # convert to numpy array
        projected_images_train = pca.transform(grey_images_train)
        projected_images_test = pca.transform(grey_images_test)
        print(f'Dimension of the projected images: {projected_images_train.shape}')

        # train a k-NN classifier on the projected images
        knn = KNN(n_neighbors=k)
        knn.fit(projected_images_train, train_labels)

        print(f'Score for s={s}: {knn.score(projected_images_train, train_labels)}')

run_PCA_KNN_ondata([50, 100, 200, 300], [5 for _ in range(5)])
