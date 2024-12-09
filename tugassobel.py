from tkinter import Image
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import convolve # type: ignore


image = imageio.imread("D:/KAMPUS NUSA PUTRA/SEMESTER 5/lena.png",mode='F')

# Fungsi untuk menerapkan operator Roberts
def roberts_edge_detection(image):
    # Definisikan kernel Roberts
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    
    # Terapkan konvolusi dengan kernel Roberts
    edge_x = convolve(image, kernel_x)
    edge_y = convolve(image, kernel_y)
    
    # Hitung magnitudo dari gradien
    edges = np.hypot(edge_x, edge_y)
    edges = edges / np.max(edges)  # Normalisasi
    return edges

# Fungsi untuk menerapkan operator Sobel
def sobel_edge_detection(image):
    # Definisikan kernel Sobel
    kernel_x = np.array([
        [1, 0, -1], 
        [2, 0, -2], 
        [1, 0, -1]
        
     ])

    kernel_y = np.array([
        [1, 2, 1], 
        [0, 0, 0],
        [-1,-2,-1]

    ])
    
    # Terapkan konvolusi dengan kernel Sobel
    edge_x = convolve(image, kernel_x)
    edge_y = convolve(image, kernel_y)
    
    # Hitung magnitudo dari gradien
    edges = np.hypot(edge_x, edge_y)
    edges = edges / np.max(edges)  # Normalisasi
    return edges

# Membaca gambar
image_path = 'path_to_your_image.jpg'  # Ganti dengan path gambar Anda
image = imageio.imread(image_path, as_gray=True)

# Deteksi tepi menggunakan Roberts dan Sobel
edges_roberts = roberts_edge_detection(image)
edges_sobel = sobel_edge_detection(image)

# Menampilkan hasil
plt.figure(figsize=(6, 6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Roberts Edge Detection')
plt.imshow(edges_roberts, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Sobel Edge Detection')
plt.imshow(edges_sobel, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()