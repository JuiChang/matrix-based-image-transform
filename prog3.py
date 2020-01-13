import numpy as np
import math
import cv2
from BlockTransform import dct, wht, dft

if __name__ == "__main__":
    img = cv2.imread("bridge.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (360, 640))
    block_size = 8

    # get size of the image
    [h, w] = img.shape

    height = h
    width = w
    h = np.float32(h)
    w = np.float32(w)

    nbh = math.ceil(h / block_size)
    nbh = np.int32(nbh)

    nbw = math.ceil(w / block_size)
    nbw = np.int32(nbw)

    # Pad the image, because sometime image size is not dividable to block size
    # get the size of padded image by multiplying block size by number of blocks in height/width

    # height of padded image
    H = block_size * nbh

    # width of padded image
    W = block_size * nbw

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

    for i in range(nbh):

        # Compute start and end row index of the block
        row_ind_1 = i * block_size
        row_ind_2 = row_ind_1 + block_size

        for j in range(nbw):
            # Compute start & end column index of the block
            col_ind_1 = j * block_size
            col_ind_2 = col_ind_1 + block_size

            block = padded_img[row_ind_1: row_ind_2, col_ind_1: col_ind_2]

            # apply 2D discrete cosine transform to the selected block
            # print("by cv2:\n", cv2.dct(block))
            # print("our dct:\n", dct(block, block_size))
            # print("our wht:\n", wht(block, block_size))
            print("our dft:\n", dft(block, block_size))
            exit()