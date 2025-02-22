import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from pytorch_msssim import ms_ssim
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Function to compute Multi-Scale SSIM (MS-SSIM) between two images
def compute_ms_ssim(img1, img2):
    """
    Compute the MS-SSIM between two PIL images.
    Both images should be in grayscale and of the same dimensions.
    """
    # Convert PIL images to tensors (values in [0, 1])
    transform = transforms.ToTensor()
    tensor1 = transform(img1).unsqueeze(0)  # Add batch dimension
    tensor2 = transform(img2).unsqueeze(0)

    # Compute MS-SSIM; data_range is set to 1.0 because the tensor values are in [0,1]
    ms_ssim_value = ms_ssim(tensor1, tensor2, data_range=1.0)
    return ms_ssim_value.item()


# Set image paths (update the path according to your environment)
image_paths = [f"./shanghaitech/{i}.jpg" for i in range(1, 14)]

# Load images: convert to grayscale and resize them
images = []
for path in image_paths:
    try:
        img = Image.open(path).convert("L")  # Convert image to grayscale
        # img = img.resize(fixed_size)  # Resize image to fixed dimensions
        images.append(img)
    except Exception as e:
        print(f"Failed to load image: {path}. Error: {e}")

# Ensure all images are loaded
if len(images) < len(image_paths):
    raise ValueError("Some images could not be loaded. Please check the file paths or formats.")

# Compute the similarity matrix using MS-SSIM
n = len(images)
similarity_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):  # Use symmetry to avoid duplicate computation
        ms_ssim_value = compute_ms_ssim(images[i], images[j])
        similarity_matrix[i, j] = ms_ssim_value
        similarity_matrix[j, i] = ms_ssim_value

# Normalize the non-diagonal elements of the similarity matrix
non_diag_indices = np.triu_indices(n, k=1)
max_val = np.max(similarity_matrix[non_diag_indices])
min_val = np.min(similarity_matrix[non_diag_indices])
for i in range(n):
    for j in range(n):
        if i != j:
            similarity_matrix[i, j] = (similarity_matrix[i, j] - min_val) / (max_val - min_val)
similarity_matrix[similarity_matrix < 0] = 0
np.fill_diagonal(similarity_matrix, 1)

# Set LaTeX font settings for Nature-style aesthetics
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Create a heatmap plot with Nature-style aesthetics
plt.figure(figsize=(12, 6))
cmap = plt.cm.viridis_r  # Reversed viridis colormap

plt.imshow(similarity_matrix, cmap=cmap, aspect='auto')
plt.xticks(ticks=np.arange(n), labels=[f"Scene {i + 1}" for i in range(n)], rotation=90, fontsize=12)
plt.yticks(ticks=np.arange(n), labels=[f"Scene {i + 1}" for i in range(n)], fontsize=12)
plt.xlabel(r"\textbf{Scene}", fontsize=14)
plt.ylabel(r"\textbf{Scene}", fontsize=14)
plt.title(r"\textbf{SSIM Similarity Matrix}", fontsize=16)
plt.colorbar(label=r"\textbf{Normalized MS-SSIM}")
plt.tight_layout()

# Save the heatmap as a high-resolution image
heatmap_path = "similarity_heatmap.png"
plt.savefig(heatmap_path, dpi=300)
plt.show()
