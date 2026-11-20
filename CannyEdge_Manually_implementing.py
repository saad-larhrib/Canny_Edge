import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image
img = cv.imread('saad.jpg')

if img is None:
    raise FileNotFoundError("Please Check the path of the image")

height, width, _ = img.shape

# 2. Converft the Img to GrayScale Manually
GrayScale = np.zeros((height, width), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        b, g, r = img[y, x]
        grayvalue = 0.2126 * float(r) + 0.7152 * float(g) + 0.0722 * float(b)
        GrayScale[y, x] = int(np.around(grayvalue))
    
# 3. Buoild 3x3 Gaussian Kernel from coordinates
sigma = 1.0

# Fixed relative coordinates around center (0, 0)
coords_3x3 = [
    (-1,  1), ( 0,  1), ( 1,  1),
    (-1,  0), ( 0,  0), ( 1,  0),
    (-1, -1), ( 0, -1), ( 1, -1)
]

# Calculate raw weights exp(-(x^2 + y^2)/(2*sigma^2))
raw_weights = [np.exp(-(dx**2 + dy**2) / (2 * sigma**2)) for dx, dy in coords_3x3]

kernel = np.array(raw_weights, dtype=np.float32).reshape((3, 3))
kernel /= np.sum(kernel)

print("Generated 3x3 Gaussian Kernel:")
print(kernel)
print("Kernel sum:", np.sum(kernel))
print(f"Kernel Shape : {kernel.shape}")

# 4. Applying the Gaussian Kernal on Img
Gaussian_filter = np.zeros((height, width), dtype=np.uint8)

kheight , kwidth = kernel.shape

for y in range(1, height-1):
    for x in range(1, width-1):
        f = GrayScale[y-1: y+2, x-1: x+2]
        res = 0
        for dy in range(kheight):
            for dx in range(kwidth):
                res += kernel[dy, dx] * f[dy, dx]
        
        Gaussian_filter[y, x] = int(np.around(res))

# 5. comparaison with my manual method and OpenCV method

opencvGrayScale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
opencvGaussianFilter = cv.GaussianBlur(
    opencvGrayScale,
    (3, 3),
    sigmaX=1
)

# 6. Show all results at once

plt.figure(figsize=(15, 10))

# 1. Original image
plt.subplot(2, 3, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# 2. Manual Grayscale
plt.subplot(2, 3, 2)
plt.imshow(GrayScale, cmap='gray')
plt.title("Manual Grayscale")
plt.axis("off")

# 3. OpenCV Grayscale
plt.subplot(2, 3, 3)
plt.imshow(opencvGrayScale, cmap='gray')
plt.title("OpenCV Grayscale")
plt.axis("off")

# 4. Manual Gaussian Filter
plt.subplot(2, 3, 4)
plt.imshow(Gaussian_filter, cmap='gray')
plt.title("Manual Gaussian Filter")
plt.axis("off")

# 5. OpenCV Gaussian Filter
plt.subplot(2, 3, 5)
plt.imshow(opencvGaussianFilter, cmap='gray')
plt.title("OpenCV Gaussian Filter")
plt.axis("off")

plt.tight_layout()
plt.show()