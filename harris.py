import cv2
import numpy as np
import copy
import sys

try:
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])
except Exception as e:
    print "MOAR arguments"
    sys.exit()

# print type(img1)
img_orig1 = copy.copy(img1)
img_orig2 = copy.copy(img2)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# must give a float32 data type input
gray1 = np.float32(gray1)
dst1 = cv2.cornerHarris(gray1, 2, 3, 0.04)
# cv2.cornerHarris(src, blockSize, ksize, k) -> dst
# src - Input single-channel 8-bit or floating-point image.
# dst - Image to store the Harris detector responses. It has the type CV_32FC1 and the same size as src .
# blockSize - Neighborhood size.
# ksize - Aperture parameter for the Sobel() operator.
# k - Harris detector free parameter.
gray2 = np.float32(gray2)
dst2 = cv2.cornerHarris(gray2, 2, 3, 0.04)
# print type(dst1)

# result is dilated for marking the corners, not important
dst1 = cv2.dilate(dst1, None)
dst2 = cv2.dilate(dst2, None)
# cv2.imshow('Harris output', dst)

# Threshold for an optimal value, it may vary depending on the image.
img1[dst1 > 0.001 * dst1.max()] = [0, 0, 255]
img2[dst2 > 0.001 * dst2.max()] = [0, 0, 255]

dst1[dst1 < 0.01 * dst1.max()] = 0
dst2[dst2 < 0.01 * dst2.max()] = 0

for i in range(0, dst1.shape[0]):
    for j in range(0, dst1.shape[1]):
        if dst1[i][j] > 0:
            x = i
            y = j
            break

for i in range(0, dst2.shape[0]):
    for j in range(0, dst2.shape[1]):
        if dst2[i][j] > 0:
            x = x - i
            y = y - j
            break
dst2 = np.roll(np.roll(dst2, x, 0), y, 1)

diff = np.sqrt(abs(dst1 ** 2 - dst2 ** 2))
diss = np.sum(diff) / (dst2.shape[0] * dst2.shape[1])
print "Dissimilarity: " + str(diss)
if diss > 100000:
    print "Images are dissimilar"
else:
    print "Images are similar"

cv2.imshow('Corners #1', img1)
cv2.imshow('Corners #2', img2)
cv2.imshow('Difference', diff)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
