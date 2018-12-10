import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

mu = 2000
sigma = 0
n_bins = 256

img = cv2.imread('The_River_10.jpg',cv2.IMREAD_GRAYSCALE)
equ = cv2.equalizeHist(img)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_equ = cv2.calcHist([equ], [0], None, [256], [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf *hist.max()/ cdf.max() # this line not necessary.
print(cdf_normalized)

fig, ax = plt.subplots(figsize=(8, 4))
i,j = np.unravel_index(hist.argmax(), hist.shape)

c = []
histScale = hist
for item in cdf_normalized:
    c.append(item)

for I in range(0, len(c)):
    histScale[I] = ((1/hist[i,j]) * hist[I])
    #print(histScale[I])

# plot the cumulative histogram
#n, bins, patches = ax.hist(histScale,n_bins,density=True,histtype='step', label='Histogram')
# Add a line showing the expected distribution.
y = []
for I in range(0, len(c)):
    s = 0
    for i in range(0, I):
        s += c[i]
    #print(s)
    s = ((1/img.size) * s)
    y.append(s)
mat = np.array(y)
print(mat[len(c)-1]+((1/256) * cdf_normalized[len(c)-1]))
#y = (1/equ.total)hist[0:10:1]
#y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
#y = y.cumsum()
#y /= y[-1]

ax.plot(mat, linewidth=1.5, label='Grayscale')

# Overlay a reversed cumulative histogram.
#ax.hist(hist, bins=bins, density=True, histtype='step', cumulative=-1,
#        label='Reversed emp.')

# tidy up the figure
ax.grid(True)
ax.legend(loc='right')
ax.set_title('Cumulative step histograms')
#plt.xlim([0, 256])
#plt.ylim([0, 256])
plt.show()
