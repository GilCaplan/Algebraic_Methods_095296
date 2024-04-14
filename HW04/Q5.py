from PIL import Image
import numpy as np
import pickle
from numpy.linalg import svd


def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res


def load_cifar10_data():
    """
    Load the CIFAR-10 data.

    returns:
    train_data: np.array, shape=(n, d), the training data
    train_labels: list, the labels of the training data
    test_data: np.array, shape=(m, d), the test data
    test_labels: list, the labels of the test data
    """

    print("Loading CIFAR-10 data...")
    train_data = None
    train_labels = []
    for i in range(1, 6):
        data_dic = unpickle('cifar-10-python/cifar-10-batches-py/data_batch_' + str(i))
        if i == 1:
            train_data = data_dic[b'data']
        else:
            train_data = np.concatenate((train_data, data_dic[b'data']), axis=0)
        train_labels += data_dic[b'labels']
    test_data_dic = unpickle('cifar-10-python/cifar-10-batches-py/test_batch')
    test_data = test_data_dic[b'data']
    test_labels = test_data_dic[b'labels']
    return train_data, train_labels, test_data, test_labels


def convert_to_grayscale(images, train=True):
    """
    Convert the given images to grayscale.

    params:
    images: np.array, shape=(n, d), the images to convert
    train: bool, whether the images are training images or not

    returns:
    np.array, shape=(n, d), the converted images
    """

    if train:
        print("Converting train images to grayscale...")
    else:
        print("Converting test images to grayscale...")
    
    flattened_images = []
    for img in images:
        img_reshaped = np.transpose(np.reshape(img, (3, 32, 32)), (1, 2, 0))
        image = Image.fromarray(img_reshaped.astype('uint8'))
        image = image.convert('L')
        flattened_image = np.array(image).flatten()
        flattened_images.append(flattened_image)
    return np.array(flattened_images)
    

def svd_decompose(images, s):
    """
    Perform SVD on the given images and return the first s principal components.

    params:
    images: np.array, shape=(d, m), the images to decompose
    s: int, the number of principal components to return

    returns:
    U: np.array, shape=(d, s), the first s principal components
    """ 

    print("Decomposing the images using SVD...")
    # make a centered matrix
    images = images - np.mean(images, axis=1)[:, np.newaxis]
    U, _, _ = svd(images, full_matrices=False)
    print("Decomposition done.")
    return U[:, :s]


def KNN(train_data, train_labels, test_data, test_labels, k_vals):
    """
    Perform KNN on the given data.

    params:
    train_data: np.array, shape=(s, n), the training data
    train_labels: list, the labels of the training data
    test_data: np.array, shape=(s, m), the test data
    test_labels: list, the labels of the test data
    k_vals: list, the values of k to be used in KNN

    returns:
    errors: dict, the number of errors for each k value
    """

    train_labels = np.array(train_labels)
    print("Running KNN...")
    # Calculate inner products
    inner_products = np.matmul(train_data.T, test_data)
    # Calculate norms
    train_norms = np.square(np.linalg.norm(train_data, axis=0))

    errors = {}
    for k in k_vals:
        errors[k] = 0
    
    for i in range(test_data.shape[1]):
        # Calculate distances
        distances = train_norms - 2*inner_products[:, i]

        for k in k_vals:
            # Find the k nearest neighbors
            nearest_neighbors = np.argsort(distances)[:k]
            # Find the labels of the k nearest neighbors
            nearest_labels = train_labels[nearest_neighbors]
            # Find the most common label
            prediction = np.bincount(nearest_labels).argmax()
            
            if prediction != test_labels[i]:
                errors[k] += 1
    
    return errors


if __name__ == '__main__':
    # Load CIFAR-10 data
    train_data, train_labels, test_data, test_labels = load_cifar10_data()
    # Convert images to grayscale, flip matrices so that each column is a flattened image
    train_data_gray = convert_to_grayscale(train_data).T
    test_data_gray = convert_to_grayscale(test_data, train=False).T

    # Center the training data
    mean_train = np.mean(train_data_gray, axis=1)
    train_data_gray_centered = (train_data_gray - mean_train[:, np.newaxis])
    test_data_gray_centered = (test_data_gray - mean_train[:, np.newaxis])

    # KNN preparation
    k_vals = [5, 10, 50, 100, 200]
    s_vals = [5, 10, 50, 100, 500, 1000]


    for s in s_vals:
        # Principal components
        Us = svd_decompose(train_data_gray_centered, s)

        # Cheap vectorization of the test and train data
        cheap_train = np.matmul(Us.T, train_data_gray_centered)
        cheap_test = np.matmul(Us.T, test_data_gray_centered)

        # Run KNN on "cheap" testing data, report testing errors
        errors = KNN(cheap_train, train_labels, cheap_test, test_labels, k_vals)
        for k in k_vals:
            print(f"Testing Error for Projected Data, s={s} and k={k} is: {errors[k]/len(test_labels)}.")

        # Run KNN on "cheap" training data, report training errors
        errors = KNN(cheap_train, train_labels, cheap_train[:, :1000], train_labels[:1000], k_vals)
        for k in k_vals:
            print(f"Training Error for Projected Data, s={s} and k={k} is: {errors[k]/1000}.")


    # Perform KNN on the original testing data, report testing errors
    errors = KNN(train_data_gray_centered, train_labels, test_data_gray_centered, test_labels, k_vals)
    for k in k_vals:
        print(f"Testing Error for Original Data, k={k} is: {errors[k]/len(test_labels)}.")

    # Perform KNN on the original training data, report training errors
    errors = KNN(train_data_gray_centered, train_labels, train_data_gray_centered[:, :1000], train_labels[:1000], k_vals)
    for k in k_vals:
        print(f"Training Error for Original Data, k={k} is: {errors[k]/1000}.")






