from PIL import Image
import numpy as np


def rank_k_approximation(A, k):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # rank-k truncation from the SVD of A
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    A_k = np.dot(U_k, np.dot(S_k, Vt_k))

    return A_k, relative_error(S, k)


def relative_error(S, k):
    return (S[k]) ** 2 / S[0] ** 2


def compress_image_to_k(path, k):
    image = Image.open(path)
    pix = np.array(image)

    noise = np.random.normal(0, 10, pix.shape)
    pix = pix + noise

    imgR = pix[:, :, 0]  # Red
    imgG = pix[:, :, 1]  # Green
    imgB = pix[:, :, 2]  # Blue
    print(pix.shape)

    error = [0, 0, 0]
    imgR_k, error[0] = rank_k_approximation(imgR, k)
    imgG_k, error[1] = rank_k_approximation(imgG, k)
    imgB_k, error[2] = rank_k_approximation(imgB, k)

    new_img = np.stack([imgR_k, imgG_k, imgB_k], axis=-1)
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    new_img = Image.fromarray(new_img)

    return new_img, error


img, error = compress_image_to_k('RedDots.jpg', 10)
print('relative error is: ', error)
img.show()

