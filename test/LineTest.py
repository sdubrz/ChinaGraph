import numpy as np


def test():
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [11, 12, 13]])
    C = np.matmul(A, A.T)
    print(C)
    (U, S, V) = np.linalg.svd(C)
    print("U = ")
    print(U)
    print("S = ")
    print(S)
    print(S.shape)
    print(S[0])
    print(S[1])

    S0 = np.zeros((4, ))
    print(S0.shape)

    S0 = S0 + S
    print(S0)


if __name__ == '__main__':
    test()
