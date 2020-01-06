import numpy as np
import math


# def generateDFTBasisBlocks(size=2):
#     basisBlocks = np.zeros((size, size, size, size))
#
#
#     return basisBlocks


# textbook p487 Eq.7-83
def dctInverseTransformationKernel(x, u, N):
    if u == 0:
        return (1 / N)**0.5 * math.cos((2 * x + 1) * u * math.pi / (2 * N))
    elif 1 <= u <= N - 1:
        return (2 / N) ** 0.5 * math.cos((2 * x + 1) * u * math.pi / (2 * N))
    else:
        print("Error from dctInverseTransformationKernel()")
        exit()


# textbook p468 Eq.7-22
def dctBasisVector(u, N):
    su = np.zeros((N, 1))
    for i in range(N):
        su[i, 0] = dctInverseTransformationKernel(i, u, N)
    return su


# textbook p468 Eq.7-24
def dctTransformationMatrix(N):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, :] = dctBasisVector(i, N).T
    return A


# textbook p470 Eq.7-35
def dct(F, N):
    A = dctTransformationMatrix(N)
    return (A.dot(F)).dot(A.T)