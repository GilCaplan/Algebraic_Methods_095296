import math
import numpy as np

A = np.array([[0,0,1/math.sqrt(2)], [1/math.sqrt(2),1/math.sqrt(2),0],[1/math.sqrt(2),-1/math.sqrt(2),0]])
B = np.array([[math.sqrt(2),0,0],[0,math.sqrt(2),0],[0,0, 1]])
print(B @ A)
# Q8
def QR_division(A):
    z, Q = Grand_Shit(A)
    R = find_R(A, Q, z)

    return Q, R


def find_R(A, Q, Z):
    R = np.array([[0 for _ in range(len(A[0]))] for _ in range(len(A[0]))], float)
    Q = np.array(Q, float)
    A = np.array(A, float)
    for i in range(len(R)):  # row
        for j in range(len(R[0])):  # col
            if i == j:
                R[i][i] = math.sqrt(sum(f ** 2 for f in Z.T[i]))
            elif i < j:
                R[i][j] = inner_mul(Q.T[i], A.T[j])
    return R


def Grand_Shit(A):
    A = np.array(A, float).T
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
    V = np.array(V, float)
    x = math.sqrt(sum(V[i] * V[i] for i in range(len(V))))
    return np.array([v_i / x for v_i in V], float)


M = [[3, 6, 8, 0, 4, 3, 1, 5, 4, 4],
 [4 ,0 ,6 ,5 ,1 ,9 ,3 ,3 ,3 ,3],
 [5 ,0 ,9 ,8 ,0 ,4 ,9 ,6 ,6 ,4],
 [0 ,7 ,6 ,9 ,2 ,5 ,5 ,5 ,3 ,4],
 [2 ,3 ,8 ,1 ,2 ,2 ,6 ,6 ,6 ,4],
 [5 ,4 ,1 ,8 ,1 ,5 ,8 ,9 ,5 ,3],
 [0 ,1 ,7 ,5 ,3 ,7 ,9 ,4 ,0 ,7],
 [2 ,9 ,2 ,8 ,3 ,4 ,8 ,2 ,2 ,5],
 [6 ,6 ,0 ,0 ,4 ,6 ,8 ,2 ,7 ,1],
 [4 ,7 ,8 ,6 ,4 ,8 ,7 ,8 ,2 ,7],
 [7 ,5 ,9 ,9 ,5 ,1 ,8 ,4 ,3 ,8],
 [2 ,4 ,9 ,2 ,9 ,4 ,0 ,7 ,0 ,8],
 [2 ,8 ,2 ,4 ,2 ,4 ,6 ,3 ,5 ,1],
 [2 ,9 ,6 ,8 ,2 ,5 ,9 ,0 ,0 ,9],
 [1 ,4 ,5 ,2 ,2 ,2 ,2 ,6 ,9 ,5]]

x = np.array([21,11,9,6,5,4,2,1,94,91,89,85,84,16,98])

np.set_printoptions(precision=2)
Q1, R1 = QR_division(M)
print(f"Q = , {Q1}, \n")
print(f"R = , {R1}, \n")
print("part 2:", Q1 @ Q1.T @ x)
# projection of x on S is equal to P @ P^T @ x
print("part 3", (np.eye(len(Q1)) - Q1 @ Q1.T) @ x)
# projection of x on S complement is (I - P @ P^T) @ x as proved in HW02 QB


Mat = np.array([[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]])
Q, R = QR_division(Mat)
print("example from prev hw: ")
print('Q= ', Q, "\nR= ", R, "\n")

