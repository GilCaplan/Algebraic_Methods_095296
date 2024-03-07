import pickle
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN


class KNNCls:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            # Calculate distances between x and all examples in the training set
            distances = [(sum(x[i] - x_train[i] for i in range(len(x))), y_train) for x_train, y_train in zip(self.X_train, self.y_train)]
            # Use np.argpartition with a custom key function to find indices of k smallest elements
            close_k_indices = np.argsort(distances)[:self.k]

            label_counts = {} 
            # (distance, label)
            for _, label in close_k_indices:
                label_counts[label] = label_counts.get(label, 0) + 1

            # Find the most common label
            predictions.append(sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[0][0])

        return np.array(predictions)


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

def KNN_1():
    pass

def run_PCA_KNN_on_data(s_vals, k_vals):
    for s, k in zip(s_vals, k_vals):
        pca = PCA(n_components=s)

        # project the images onto the first s principal components
        pca.fit(grey_images_train)
        projected_images_train = pca.transform(grey_images_train)
        print(f'Dimension of the projected images: {projected_images_train.shape}')

        # train a k-NN classifier on the projected images
        #knn = KNN(n_neighbors=k)
        knn = KNNCls()
        knn.fit(projected_images_train, train_labels)

        # Predict on the training set using transformed data
        pred_train = knn.predict(projected_images_train)

        # Calculate training accuracy
        train_accuracy = np.mean(pred_train == train_labels)
        print(f"Training Accuracy for s={s}, k={k}: {train_accuracy}")

        # Uncomment the following lines if you want to calculate test accuracy
        projected_images_test = pca.transform(grey_images_test)
        projected_images_test = pca.transform(grey_images_test)
        pred_test = knn.predict(projected_images_test)
        accuracy = np.mean(pred_test == test_labels)
        print(f"Test Accuracy for s={s}, k={k}: {accuracy}")


# Assuming grey_images_train, train_labels, grey_images_test, and test_labels are defined
run_PCA_KNN_on_data(s_vals := [40, 50], [6 for _ in range(len(s_vals))])
