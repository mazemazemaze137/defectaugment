import cv2
import matplotlib.pyplot as plt

orig = cv2.imread("data/processed/mock_defects/Crack/crack_0003.png", cv2.IMREAD_GRAYSCALE)
aug = cv2.imread("results/traditional/Crack_crack_0003_png_aug0012.png", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.imshow(orig, cmap='gray'); plt.title("Original")
plt.subplot(1, 2, 2); plt.imshow(aug, cmap='gray'); plt.title("Augmented")
plt.show()
