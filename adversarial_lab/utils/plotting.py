from typing import Any

import numpy as np
import matplotlib.pyplot as plt


def plot_images_and_noise(image, noise):
    if image.shape[0] == 1:
        image = image[0]

    if noise.shape[0] == 1:
        noise = noise[0]

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    im1 = axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    noisy_image = image + noise
    im2 = axes[1].imshow(noisy_image)
    axes[1].set_title('Image with Noise Applied')
    axes[1].axis('off')

    normalized_noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
    im3 = axes[2].imshow(normalized_noise)
    axes[2].set_title('Normalized Noise')
    axes[2].axis('off')

    plt.show()
