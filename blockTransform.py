import numpy as np
import math


# textbook p487 Eq.7-83
def dctInverseTransformationKernel(x, u, N):
    if u == 0:
        return (1 / N)**0.5 * math.cos((2 * x + 1) * u * math.pi / (2 * N))
    elif 1 <= u <= N - 1:
        return (2 / N) ** 0.5 * math.cos((2 * x + 1) * u * math.pi / (2 * N))
    else:
        print("Error from dctInverseTransformationKernel()")
        exit()


# textbook p498 Eq.7-102
def whtInverseTransformationKernel(x, u, N):

    ### calculate the power
    n = int(round(math.log2(N)))
    power = 0

    # _x means that the input of this function mentioned in the textbook is fix to x
    b_x = [int(i) for i in bin(x)[2:]]
    b_x.reverse()
    while len(b_x) < n:
        b_x.append(0)

    b_u = [int(i) for i in bin(u)[2:]]
    b_u.reverse()
    while len(b_u) < n:
        b_u.append(0)

    p_u = [0] * n
    p_u[0] = b_u[n - 1]
    for i in range(1, n):
        p_u[i] = b_u[n - i] + b_u[n - i - 1]

    for i in range(n):
        power += b_x[i] * p_u[i]
    power = power % 2
    return (1 / N)**0.5 * (-1)**power


# textbook p476 Eq.7-56
def dftInverseTransformationKernel(x, u, N):
    p = 2 * math.pi * u * x / N
    return 1 / N**0.5 * (math.cos(p) + 1j * math.sin(p))


# textbook p468 Eq.7-22
def basisVector(u, N, tf):
    if tf == 'dct':
        su = np.zeros((N, 1))
        for i in range(N):
            su[i, 0] = dctInverseTransformationKernel(i, u, N)
    elif tf == 'wht':
        su = np.zeros((N, 1))
        for i in range(N):
            su[i, 0] = whtInverseTransformationKernel(i, u, N)
    elif tf == 'dft':
        su = np.zeros((N, 1), dtype=complex)
        for i in range(N):
            su[i, 0] = dftInverseTransformationKernel(i, u, N)
    return su


# textbook p468 Eq.7-24
def transformationMatrix(N, tf='dct'):
    if tf in ['dct', 'wht']:
        A = np.zeros((N, N))
    elif tf == 'dft':
        A = np.zeros((N, N), dtype=complex)

    for i in range(N):
        A[i, :] = basisVector(i, N, tf).T
    if tf == 'dft':
        A = np.conj(A)
        # should meet textbook p485 Fig.7.7(a) when N, the blockSize, is 8
        # print("8**0.5 * dft matrix\n", 8**0.5 * A)
    return A


# textbook p470 Eq.7-35
def dct(F, N):
    A = transformationMatrix(N, 'dct')
    return (A.dot(F)).dot(A.T)


# textbook p470 Eq.7-35
def wht(F, N):
    A = transformationMatrix(N, 'wht')
    return (A.dot(F)).dot(A.T)


# textbook p473 Eq.7-41
def dft(F, N):
    A = transformationMatrix(N, 'dft')
    # not sure if the textbook has no mistake
    # Astar = np.conj(A)
    # return (Astar.dot(F)).dot(Astar.T)
    return (A.dot(F)).dot(A.T)


# the combination version of the three functions above
def transform(F, N, tf="dct"):
    A = transformationMatrix(N, tf)
    return (A.dot(F)).dot(A.T)


# Set10 slide p24
def generateDctBasis(N):
    basis = np.zeros((N, N, N, N)) # (x, y, u, v)
    tmp1D = np.zeros((N, N))
    for x in range(N):
        for u in range(N):
            tmp1D[x, u] = dctInverseTransformationKernel(x, u, N)
    for x in range(N):
        for y in range(N):
            for u in range(N):
                for v in range(N):
                    basis[x, y, u, v] = tmp1D[x, u] * tmp1D[y, v]
    return basis


# textbook p499 Eq.7-107 "separable"
def generateWhtBasis(N):
    basis = np.zeros((N, N, N, N)) # (x, y, u, v)
    tmp1D = np.zeros((N, N))
    for x in range(N):
        for u in range(N):
            tmp1D[x, u] = whtInverseTransformationKernel(x, u, N)
    for x in range(N):
        for y in range(N):
            for u in range(N):
                for v in range(N):
                    basis[x, y, u, v] = tmp1D[x, u] * tmp1D[y, v]
    return basis


def generateDftBasis(N):
    basis = np.zeros((N, N, N, N), dtype=complex) # (x, y, u, v)
    tmp1D = np.zeros((N, N), dtype=complex)
    for x in range(N):
        for u in range(N):
            tmp1D[x, u] = dftInverseTransformationKernel(x, u, N)
            # print(tmp1D[x, u])
    for x in range(N):
        for y in range(N):
            for u in range(N):
                for v in range(N):
                    basis[x, y, u, v] = tmp1D[x, u] * tmp1D[y, v]
    return basis


# textbook p470 Eq.7-36
def reconstructDct(subImgCof, N):
    A = transformationMatrix(N, 'dct')
    return A.T.dot(subImgCof).dot(A)


# textbook p470 Eq.7-36
def reconstructWht(subImgCof, N):
    A = transformationMatrix(N, 'wht')
    return A.T.dot(subImgCof).dot(A)


# textbook p473 Eq.7-42
def reconstructDft(subImgCof, N):
    A = transformationMatrix(N, 'dft')
    A = np.conj(A)
    return A.T.dot(subImgCof).dot(A)


# the combination version of the three functions above
def reconstruct(subImgCof, N, tf="dct"):
    A = transformationMatrix(N, tf)
    if tf == "dft":
        A = np.conj(A)
    return A.T.dot(subImgCof).dot(A)
