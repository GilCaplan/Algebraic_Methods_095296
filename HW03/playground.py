import math
import numpy as np


def QR_division(A):
    z, Q = Grand_Shit(A)
    R = find_R(A, Q, z)

    return Q, R


def find_R(A, Q, Z):
    R = np.array([[0 for _ in range(len(A[0]))] for _ in range(len(A[0]))])

    for i in range(len(R)):  # row
        for j in range(len(R[0])):  # col
            if i == j:
                R[i][i] = math.sqrt(sum(x * x for x in Z.T[i]))
            elif i < j:
                R[i][j] = inner_mul(Q.T[i], A.T[j])
    return R


def Grand_Shit(A):
    A = A.T
    z = np.array([[0 for _ in range(len(A[0]))] for _ in range(len(A))], float)
    z[0] = A[0]

    for i in range(1, len(A)):
        series_sum = sum(cal_proj(z[j], A[i]) for j in range(i))
        z[i] = A[i] - series_sum

    W = np.array([[0 for _ in range(len(z[0]))] for _ in range(len(z))], float)
    for i in range(len(W)):
        W[i] = normalize(z[i])

    return z.T, W.T


def cal_proj(V1, V2):
    return (inner_mul(V1, V2) / inner_mul(V1, V1)) * V1


def inner_mul(V1, V2):
    return sum(V1[i] * V2[i] for i in range(len(V1)))


def normalize(V):
    V = np.array(V)
    x = math.sqrt(sum(V[i] * V[i] for i in range(len(V))))
    return np.array([v_i / x for v_i in V])


def QR_division(A):
    z, Q = Grand_Shit(A)
    R = find_R(A, Q, z)

    return Q, R


def find_R(A, Q, Z):
    R = np.array([[0 for _ in range(len(A[0]))] for _ in range(len(A[0]))])

    for i in range(len(R)):  # row
        for j in range(len(R[0])):  # col
            if i == j:
                R[i][i] = math.sqrt(sum(x * x for x in Z.T[i]))
            elif i < j:
                R[i][j] = inner_mul(Q.T[i], A.T[j])
    return R


def Grand_Shit(A):
    A = A.T
    z = np.array([[0 for _ in range(len(A[0]))] for _ in range(len(A))], float)
    z[0] = A[0]

    for i in range(1, len(A)):
        series_sum = sum(cal_proj(z[j], A[i]) for j in range(i))
        z[i] = A[i] - series_sum

    W = np.array([normalize(row) for row in z], float)

    return z.T, W.T


def cal_proj(V1, V2):
    return (inner_mul(V1, V2) / inner_mul(V1, V1)) * V1


def inner_mul(V1, V2):
    return sum(V1[i] * V2[i] for i in range(len(V1)))


def normalize(V):
    V = np.array(V)
    x = math.sqrt(sum(V[i] * V[i] for i in range(len(V))))
    return np.array([v_i / x for v_i in V])


Mat = np.array([[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]])
Q, R = QR_division(Mat)

print(Q)
print(R)