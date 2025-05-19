import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import re
import io
from natsort import natsorted  # Ensures proper numerical sorting

results = "results"
output_path = os.path.join(results, "pca_results")
SAVE_DIR = os.path.join(output_path, "pca_result_activation_layer")
INPUT_DIR = os.path.join(results, "validation_results", "result_activation_17_d4")
IMG_SIZE = (512, 512)
NUM_EIGENFACES = 64
N_FACES_OUT = 10


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def read_images(folder_path):
    """Reads images, converts them to grayscale, and resizes them."""
    filenames = natsorted(os.listdir(folder_path))
    if not filenames:
        return None, None, None

    match = re.match(r"(.+?)_channel", filenames[0])
    if not match:
        return None, None, None

    img_name = match.group(1)
    ori_image_path = os.path.join("Data/test/image", f"{img_name}.png")
    ori_image = cv2.imread(ori_image_path, cv2.IMREAD_UNCHANGED)
    if ori_image is None:
        return None, None, None
    ori_resized = cv2.resize(ori_image, IMG_SIZE)

    images = [cv2.resize(cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_GRAYSCALE), IMG_SIZE)
              for f in filenames if cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_GRAYSCALE) is not None]
    
    return np.array(images, dtype=np.float32), ori_resized, img_name


def perform_pca(images, num_components):
    """Performs PCA using eigen decomposition."""
    num_images, num_pixels = images.shape[0], IMG_SIZE[0] * IMG_SIZE[1]
    data_matrix = images.reshape(num_images, num_pixels)
    mean_face = np.mean(data_matrix, axis=0)
    centered_matrix = data_matrix - mean_face
    
    covariance_matrix = np.dot(centered_matrix, centered_matrix.T) / num_images
    eigenvalues, eigenvectors_small = np.linalg.eigh(covariance_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors_small = eigenvalues[idx], eigenvectors_small[:, idx]
    
    eigenvectors = np.dot(centered_matrix.T, eigenvectors_small)
    norms = np.linalg.norm(eigenvectors, axis=0)
    norms[norms == 0] = 1  # Avoid division by zero
    eigenvectors /= norms
    
    eigenfaces = [eigenvectors[:, i].reshape(IMG_SIZE) for i in range(num_components)]
    
    return mean_face.reshape(IMG_SIZE), eigenfaces, eigenvalues[:num_components], eigenvectors[:, :num_components]


def save_plot(eigenvalues, log_scale, ignore_first, ignore_last, cur_image):
    """Generates and saves an eigenvalue plot with a legend at the top right."""
    ev = np.log(eigenvalues) if log_scale else eigenvalues
    ev = ev[1:] if ignore_first else ev
    ev = ev[:-1] if ignore_last else ev

    x_values = list(range(1, len(ev) + 1))
    
    plt.figure(figsize=(15, 8), dpi=1000)
    plt.scatter(x_values, ev, color='blue', s=20, label="Eigenvalues")
    
    plt.xlabel("Channel Number")
    plt.ylabel("Log Eigenvalues" if log_scale else "Eigenvalues")
    plt.xticks(x_values, rotation=90)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.legend(loc="upper right", fontsize=12)  # Legend at top right
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    img_buf.seek(0)
    
    return img_buf


def generate_images(images, original_img, mean_face, title, cols=4, top_n=True):
    """Displays original, mean, and eigenfaces in a grid."""
    images = images[:N_FACES_OUT] if top_n else images[-N_FACES_OUT:]
    total_images = len(images) + 2
    rows = (total_images + cols - 1) // cols
    
    plt.figure(figsize=(cols * 3, rows * 3), dpi=300)
    
    for idx, (img, label) in enumerate([(original_img, "Original"), (mean_face, "Mean Face")], start=1):
        plt.subplot(rows, cols, idx)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) if idx == 1 else img, cmap="gray")
        plt.axis("off")
        plt.title(label)
    
    for i, img in enumerate(images, start=3):
        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"Eigenface {i-2}")
    
    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    img_buf.seek(0)
    return img_buf


def save_image(image, save_path):
    with open(save_path, "wb") as f:
        f.write(image.getbuffer())

def save_plot_2(eigenvalues, log_scale, ignore_first, ignore_last):
    """Generates and saves an eigenvalue scatter plot with an adaptive x-axis."""
    ev = np.log(eigenvalues) if log_scale else eigenvalues
    ev = ev[2:] if ignore_first else ev
    ev = ev[:-1] if ignore_last else ev

    x_values = np.arange(1, len(ev) + 1)
    
    plt.figure(figsize=(12, 6), dpi=300)
    plt.scatter(x_values, ev, color='blue', s=30, label="Eigenvalues")
    
    plt.xlabel("Channel Number", fontsize=18)
    plt.ylabel("Log Eigenvalues" if log_scale else "Eigenvalues", fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.6)

    # **Adaptive X-Ticks:**
    num_xticks = min(len(x_values), 7)  # Limit number of xticks
    xtick_positions = np.linspace(x_values[0], x_values[-1], num_xticks, dtype=int)
    plt.xticks(xtick_positions, fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc="upper right", fontsize=16)  # Legend at top right

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    img_buf.seek(0)
    
    return img_buf


if __name__ == "__main__":
    create_dir(SAVE_DIR)
    hidden_layers = [f.path.replace("\\", "/") for f in os.scandir(INPUT_DIR) if f.is_dir()]
    
    for folder in hidden_layers:
        images, original_image, cur_out = read_images(folder)
        if images is None:
            print(f"Skipping {folder}: No valid images found.")
            continue
        
        print(f"Processing {len(images)} images from {folder}")
        mean_face, eigenfaces, eigenvalues, _ = perform_pca(images, NUM_EIGENFACES)
        save_directory = os.path.join(SAVE_DIR, cur_out)
        create_dir(save_directory)
        
        np.savetxt(os.path.join(save_directory, f"{cur_out}_ev.csv"), eigenvalues, delimiter=",")
        
        save_image(generate_images(eigenfaces, original_image, mean_face, title="", cols=4, top_n=True),
                   os.path.join(save_directory, f"{cur_out}_top_{N_FACES_OUT}.png"))
        save_image(generate_images(eigenfaces, original_image, mean_face, title="", cols=4, top_n=False),
                   os.path.join(save_directory, f"{cur_out}_last_{N_FACES_OUT}.png"))
        save_image(save_plot_2(eigenvalues, False, True, True),
                   os.path.join(save_directory, f"{cur_out}_graph_eigenVal_vs_channel.png"))