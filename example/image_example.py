# Author: Dawid Laszuk
# Last update: 7/07/2017
import numpy as np
import pylab as plt
from PyEMD import EMD2D

# Generate image
print("Generating image... ", end="")
rows, cols = 1024, 1024
row_scale, col_scale = 256, 256
x = np.arange(rows) / float(row_scale)
y = np.arange(cols).reshape((-1, 1)) / float(col_scale)

pi2 = 2 * np.pi
img = np.zeros((rows, cols))
img = img + np.sin(2 * pi2 * x) * np.cos(y * 4 * pi2 + 4 * x * pi2)
img = img + 3 * np.sin(2 * pi2 * x) + 2
img = img + 5 * x * y + 2 * (y - 0.2) * y
print("Done")

# Perform decomposition
print("Performing decomposition... ", end="")
emd2d = EMD2D()
# emd2d.FIXE_H = 5
IMFs = emd2d.emd(img, max_imf=4)
imfNo = IMFs.shape[0]
print("Done")

print("Plotting results... ", end="")

# Save image for preview
plt.figure(figsize=(4, 4 * (imfNo + 1)))
plt.subplot(imfNo + 1, 1, 1)
plt.imshow(img)
plt.colorbar()
plt.title("Input image")

# Save reconstruction
for n, imf in enumerate(IMFs):
    plt.subplot(imfNo + 1, 1, n + 2)
    plt.imshow(imf)
    plt.colorbar()
    plt.title("IMF %i" % (n + 1))

plt.savefig("image_decomp")
print("Done")
