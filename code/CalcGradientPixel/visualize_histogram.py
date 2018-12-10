import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('The_River_10.jpg',cv2.IMREAD_GRAYSCALE)
equ = cv2.equalizeHist(img)

# Construct a grayscale histogram
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_equ = cv2.calcHist([equ], [0], None, [256], [0, 256])
print(type(hist))
# Plot the histogram
fig = plt.figure()
fig.suptitle("Grayscale Histograms", fontsize=16)

plt.subplot("221")
plt.imshow(img, 'gray')

plt.subplot("222")
plt.imshow(equ, 'gray')

og = plt.subplot("223")
og.set_title("Orignial Histogram")
og.plot(hist)

plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.xlim([0, 256])
plt.ylim([0, 5000])

nw = plt.subplot("224")
nw.set_title("Histogram Equalization")
nw.plot(hist_equ)

plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.xlim([0, 256])
plt.ylim([0, 5000])
plt.show()
