from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def rank_k_approximation(A, k):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # rank-k truncation from the SVD of A
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    # print('rank', np.count_nonzero(S))

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


errR, errG, errB = [], [], []
k_s = [5, 20, 50, 70, 80, 100, 200]
for path in [('GilZak.jpg', k) for k in k_s]:
    img, error = compress_image_to_k(path[0], path[1])
    errR = errR + [(path[1], error[0])]
    errG = errG + [(path[1], error[1])]
    errB = errB + [(path[1], error[2])]
    print(f'for k = {path[1]} the relative error is: {error}')
    img.save(f'GilZak_k={path[1]}.jpg')



def draw_graph(data_points, l):
    # Unpack the data points into separate lists for x and y coordinates
    k_values, error_values = zip(*data_points)

    # Plotting the graph
    plt.plot(k_values, error_values, marker='o', linestyle='-', color=l, label=f'Graph {l}')

    # Adding labels and title
    plt.xlabel('k Values')
    plt.ylabel('Error Values')
    plt.title('Error vs. k Graph')

    # Display the graph
    #plt.show()
    plt.savefig(f'k_errorGraph{l}.png')


draw_graph(errR, 'r')
draw_graph(errG, 'g')
draw_graph(errB, 'b')





