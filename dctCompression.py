import numpy as np
import math
import cv2
from BlockTransform import dct, generateDctBasis, reconstructDct, coefficientsVariance

if __name__ == "__main__":
    blockSize = 8

    img = cv2.imread("bridge_hd.jpg", cv2.IMREAD_GRAYSCALE)
    # print(img.shape) # (2160, 3840)
    img = cv2.resize(img, (640, 320))
    # print(img.shape) # (320, 640)
    cv2.imwrite("bridge.jpg", img)

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

    # Pad the image, because sometime image size is not dividable to block size
    # get the size of padded image by multiplying block size by number of blocks in height/width

    # height of padded image
    H = blockSize * nbh

    # width of padded image
    W = blockSize * nbw

    # create a numpy zero matrix with size of H,W
    padded_img = np.zeros((H, W))

    # copy the values of img into padded_img[0:h,0:w]
    # for i in range(height):
    #         for j in range(width):
    #                 pixel = img[i,j]
    #                 padded_img[i,j] = pixel

    # or this other way here
    padded_img[0:height, 0:width] = img[0:height, 0:width]

    # cv2.imwrite('uncompressed.bmp', np.uint8(padded_img))

    # start encoding:
    # divide image into block size by block size (here: 8-by-8) blocks
    # To each block apply 2D discrete cosine transform
    # reorder DCT coefficients in zig-zag order
    # reshaped it back to block size by block size (here: 8-by-8)

    # the last two dimension of imgCof are corresponding to u, v, respectively
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

            # apply 2D discrete cosine transform to the selected block
            # print("by cv2:\n", cv2.dct(block))
            # print("our dct:\n", dct(block, blockSize))
            # print("our wht:\n", wht(block, blockSize))
            # print("our dft:\n", dft(block, blockSize))
            # exit()
            imgCof[i, j] = dct(block, blockSize)

    ### information packing ability
    print("coefficient variances across sub-images:\n", np.var(imgCof, axis=(0, 1)))

    ### quantization
    # 1. Keep only the first k coefficients
    # keepSize = 8
    # mask = np.add(*np.indices((blockSize, blockSize)))
    # mask = mask < keepSize
    # imgCof = imgCof * mask

    # 2. Keep only the coefficients with the k largest coefficients
    # k = 32
    # # count = 0
    # for i in range(nbh):
    #     for j in range(nbw):
    #         thres = np.sort(np.absolute(imgCof[i, j]).flatten())[-k]
    #         mask = np.absolute(imgCof[i, j]) >= thres
    #         # if count < 5:
    #         #     print(imgCof[i, j])
    #         #     print(mask)
    #         #     count += 1
    #         imgCof[i, j] = imgCof[i, j] * mask

    # 3. Distribute a fixed number of bits to all the coefficients (me: at the same position)
    # according to the logarithm of coefficient variances
    # also mentioned the mail from teacher

    totalBits = 1280
    qi = np.var(imgCof, axis=(0, 1))
    ni = np.round(totalBits * qi / qi.sum()) # ni.shape: (blockSize, blockSize)
    print(ni)
    # print(ni.sum())
    # yet: the case if ni.sum() > totalBits

    print("before:\n", imgCof[:, :, -1, -1].mean())
    print("before:\n", imgCof[:, :, -1, -1].max())
    print("before:\n", imgCof[:, :, -1, -1].min())

    for i in range(blockSize):
        for j in range(blockSize):
            # cofMin = imgCof[:, :, i, j].min()
            # cofMax = imgCof[:, :, i, j].max()
            # cofRange = cofMax - cofMin
            # if ni[i, j] == 0:
            #     imgCof[:, :, i, j] = 0
            #     continue
            # intvlWidth = cofRange / 2**ni[i, j]
            # imgCof[:, :, i, j] = cofMin + intvlWidth * ((imgCof[:, :, i, j] - cofMin) // intvlWidth + 0.5)

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

    print("after:\n", imgCof[:, :, -1, -1].mean())
    print("after:\n", imgCof[:, :, -1, -1].max())
    print("after:\n", imgCof[:, :, -1, -1].min())

    ### reconstruction
    reconsImg = np.zeros((H, W))

    # basis = generateDctBasis(blockSize)

    # for i in range(nbh):
    #
    #     # Compute start and end row index of the block
    #     row_ind_1 = i * blockSize
    #     row_ind_2 = row_ind_1 + blockSize
    #
    #     for j in range(nbw):
    #         # Compute start & end column index of the block
    #         col_ind_1 = j * blockSize
    #         col_ind_2 = col_ind_1 + blockSize
    #
    #         for u in range(blockSize):
    #             for v in range(blockSize):
    #                 reconsImg[row_ind_1: row_ind_2, col_ind_1: col_ind_2] += \
    #                     imgCof[i, j, u, v] * basis[:, :, u, v]

    for i in range(nbh):

        # Compute start and end row index of the block
        row_ind_1 = i * blockSize
        row_ind_2 = row_ind_1 + blockSize

        for j in range(nbw):
            # Compute start & end column index of the block
            col_ind_1 = j * blockSize
            col_ind_2 = col_ind_1 + blockSize

            reconsImg[row_ind_1: row_ind_2, col_ind_1: col_ind_2] += \
                reconstructDct(imgCof[i, j], blockSize)

    cv2.imwrite("reconsImg.bmp", reconsImg)


    #### fidelity

    # test
    # reconsImg = reconsImg * 0

    errMap = reconsImg[0:height, 0:width] - img
    eRMS = (1 / (height * width)) * np.sum(errMap**2)
    print("eRMS:", eRMS)
    snr = np.sum(reconsImg[0:height, 0:width]**2) / np.sum(errMap**2)
    print("SNR:", snr)