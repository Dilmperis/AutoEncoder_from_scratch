import os
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image

save_dir = "mnist_images"
os.makedirs(save_dir, exist_ok=True)

# Download MNIST
mnist = datasets.MNIST(root=".", train=True, download=True)

# Save images
for i, (img, label) in enumerate(mnist):
    img_path = os.path.join(save_dir, f"train_{i}_label_{label}.png")
    img.save(img_path)

mnist_test = datasets.MNIST(root=".", train=False, download=True)
for i, (img, label) in enumerate(mnist_test):
    img_path = os.path.join(save_dir, f"test_{i}_label_{label}.png")
    img.save(img_path)


