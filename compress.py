import numpy as np
import math
import cv2
from blockTransform import reconstruct, transform
import argparse
import os
import pickle


if __name__ == "__main__":
    # reference: https://docs.python.org/2/howto/argparse.html#introducing-optional-arguments
    parser = argparse.ArgumentParser()
    # positional argument
    parser.add_argument('input')
    # optional arguments
    parser.add_argument('-t', '--transform', default='dct')
    parser.add_argument('-b', '--blockSize', type=int, default=8)
    parser.add_argument('-q', '--quantizeId', type=int, default=1)
    parser.add_argument('-p', '--quantizePara', type=int, default=8)
    args = parser.parse_args()

    inputNameNoExt = os.path.splitext(args.input)[0]
    inputExt = os.path.splitext(args.input)[1]

    resizeFolder = 'resize'
    resizeName = inputNameNoExt + '_resize' + inputExt
    resizePath = os.path.join(resizeFolder, resizeName)
    outputFolder = 'output'
    outputName = inputNameNoExt + '_' + args.transform + '_' + str(args.blockSize) + \
                 '_' + str(args.quantizeId) + '_' + str(args.quantizePara) + '.bmp'
    outputPath = os.path.join(outputFolder, outputName)

    blockSize = args.blockSize

    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    # print(img.shape) # (2160, 3840)
    img = cv2.resize(img, (640, 320))
    # print(img.shape) # (320, 640)
    cv2.imwrite(resizePath, img)

    # get size of the image
    (h, w) = img.shape

    height = h
    width = w
    h = np.float32(h)
    w = np.float32(w)

    nbh = math.ceil(h / blockSize)
    nbh = np.int32(nbh)

    nbw = math.ceil(w / blockSize)
    nbw = np.int32(nbw)

    ### Pad the image
    # height of padded image
    H = blockSize * nbh
    # width of padded image
    W = blockSize * nbw
    padded_img = np.zeros((H, W))
    padded_img[0:height, 0:width] = img[0:height, 0:width]

    # the last two dimension of imgCof are corresponding to u, v, respectively
    if args.transform == 'dft':
        imgCof = np.zeros((nbh, nbw, blockSize, blockSize), dtype=complex)
    else:
        imgCof = np.zeros((nbh, nbw, blockSize, blockSize))
    print("nbh:", nbh, " nbw:", nbw)

    for i in range(nbh):
        # Compute start and end row index of the block
        row_ind_1 = i * blockSize
        row_ind_2 = row_ind_1 + blockSize
        for j in range(nbw):
            # Compute start & end column index of the block
            col_ind_1 = j * blockSize
            col_ind_2 = col_ind_1 + blockSize

            block = padded_img[row_ind_1: row_ind_2, col_ind_1: col_ind_2]

            imgCof[i, j] = transform(block, blockSize, args.transform)

    ### information packing ability
    # print(imgCof[0, 0])
    # print(imgCof[1, 1])
    # print(imgCof[2, 2])
    # print("mean:\n", np.mean(imgCof, axis=(0, 1)))
    cofVar = np.var(imgCof, axis=(0, 1))
    # print("coefficient variances across sub-images:\n", cofVar)
    # pickle.dump(cofVar, open("dft.pkl", "wb"))

    # textbook p581
    mask = np.add(*np.indices((blockSize, blockSize)))
    mask = mask >= blockSize
    ems = (cofVar * mask).sum()
    print("ems:", ems)

    ### quantization
    if args.quantizeId == 0:
        pass
    elif args.quantizeId == 1:
        # Keep only the first k coefficients
        keepSize = args.quantizePara
        mask = np.add(*np.indices((blockSize, blockSize)))
        mask = mask < keepSize
        imgCof = imgCof * mask

    elif args.quantizeId == 2:

        # Keep only the coefficients with the k largest coefficients
        k = args.quantizePara
        inn = list()
        for i in range(nbh):
            for j in range(nbw):
                thres = np.sort(np.absolute(imgCof[i, j]).flatten())[-k]
                # print(imgCof[i, j, 5, 5], np.absolute(imgCof[i, j, 5, 5]))
                mask = np.absolute(imgCof[i, j]) >= thres
                imgCof[i, j] = imgCof[i, j] * mask

                mask2 = np.add(*np.indices((blockSize, blockSize)))
                mask2 = mask2 < blockSize
                inn.append((mask * mask2).sum())
        print((sum(inn) / len(inn)) / blockSize**2)

    elif args.quantizeId == 3:

        # Distribute a fixed number of bits to all the coefficients (me: at the same position)
        # according to the logarithm of coefficient variances
        # also mentioned the mail from teacher

        totalBits = args.quantizePara
        qi = np.var(imgCof, axis=(0, 1))
        ni = np.round(totalBits * qi / qi.sum()) # ni.shape: (blockSize, blockSize)
        print(ni)
        # print(ni.sum())
        # yet: the case if ni.sum() > totalBits

        # print("before:\n", imgCof[:, :, -1, -1].mean())
        # print(imgCof[:, :, -1, -1].max())
        # print(imgCof[:, :, -1, -1].min())

        # count = 0

        if args.transform == 'dft':
            for i in range(blockSize):
                for j in range(blockSize):

                    cofMin = np.percentile(imgCof.real[:, :, i, j], 5)
                    cofMax = np.percentile(imgCof.real[:, :, i, j], 95)
                    cofRange = cofMax - cofMin
                    if ni[i, j] == 0:
                        imgCof.real[:, :, i, j] = 0
                        continue
                    intvlWidth = cofRange / (2 ** ni[i, j] / 2)
                    tmpCof = imgCof.real[:, :, i, j].copy()
                    tmpCof[tmpCof > cofMax] = cofMax
                    tmpCof[tmpCof < cofMin] = cofMin
                    imgCof.real[:, :, i, j] = cofMin + intvlWidth * ((tmpCof - cofMin) // intvlWidth + 0.5)

                    cofMin = np.percentile(imgCof.imag[:, :, i, j], 5)
                    cofMax = np.percentile(imgCof.imag[:, :, i, j], 95)
                    cofRange = cofMax - cofMin
                    if ni[i, j] == 0 or (i == 0 and j == 0):
                        imgCof.imag[:, :, i, j] = 0
                        continue
                    intvlWidth = cofRange / (2 ** ni[i, j] / 2)
                    if i == 0 and j == 0:
                        print('intvlWidth:', intvlWidth)
                    tmpCof = imgCof.imag[:, :, i, j].copy()
                    tmpCof[tmpCof > cofMax] = cofMax
                    tmpCof[tmpCof < cofMin] = cofMin
                    imgCof.imag[:, :, i, j] = cofMin + intvlWidth * ((tmpCof - cofMin) // intvlWidth + 0.5)

        else:
            for i in range(blockSize):
                for j in range(blockSize):

                    cofMin = np.percentile(imgCof[:, :, i, j], 5)
                    cofMax = np.percentile(imgCof[:, :, i, j], 95)
                    cofRange = cofMax - cofMin
                    if ni[i, j] == 0:
                        imgCof[:, :, i, j] = 0
                        continue
                    intvlWidth = cofRange / 2 ** ni[i, j]
                    tmpCof = imgCof[:, :, i, j].copy()
                    tmpCof[tmpCof > cofMax] = cofMax
                    tmpCof[tmpCof < cofMin] = cofMin
                    imgCof[:, :, i, j] = cofMin + intvlWidth * ((tmpCof - cofMin) // intvlWidth + 0.5)

        # print("after:\n", imgCof[:, :, -1, -1].mean())
        # print(imgCof[:, :, -1, -1].max())
        # print(imgCof[:, :, -1, -1].min())

    ### reconstruction
    reconsImg = np.zeros((H, W))

    for i in range(nbh):

        # Compute start and end row index of the block
        row_ind_1 = i * blockSize
        row_ind_2 = row_ind_1 + blockSize

        for j in range(nbw):
            # Compute start & end column index of the block
            col_ind_1 = j * blockSize
            col_ind_2 = col_ind_1 + blockSize

            if args.transform == 'dft':
                # print(imgCof[i, j])
                reconsSubImg = reconstruct(imgCof[i, j], blockSize, 'dft')
                # print(np.imag(reconsSubImg))

                reconsImg[row_ind_1: row_ind_2, col_ind_1: col_ind_2] += np.real(reconsSubImg)
                if args.quantizeId == 0 and np.sum(np.imag(reconsSubImg) > 1) > 0:
                    print("unexpected sub-image reconstruction")
                    print('np.imag(reconsSubImg).max():', np.imag(reconsSubImg).max())
                    exit()
            else:
                reconsImg[row_ind_1: row_ind_2, col_ind_1: col_ind_2] += \
                    reconstruct(imgCof[i, j], blockSize, args.transform)

    cv2.imwrite(outputPath, reconsImg)

    #### fidelity

    # test
    # reconsImg = reconsImg * 0

    errMap = reconsImg[0:height, 0:width] - img
    eRMS = (1 / (height * width)) * np.sum(errMap**2)
    print("eRMS:", eRMS)
    snr = np.sum(reconsImg[0:height, 0:width]**2) / np.sum(errMap**2)
    print("SNR:", snr)
