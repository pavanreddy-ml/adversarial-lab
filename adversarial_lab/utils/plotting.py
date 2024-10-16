from typing import Any

import numpy as np
import matplotlib.pyplot as plt


def plot_images_and_noise(image, noise, noise_generator):
    if len(image.shape) == 4:
        image_to_plot = image[0]
        noise_to_plot = noise[0]
    else:
        image_to_plot = image
        noise_to_plot = noise

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    im1 = axes[0].imshow(image_to_plot)
    axes[0].set_title('Image')
    axes[0].axis('off')

    noisy_image = noise_generator.apply_noise(image_to_plot, noise_to_plot)
    im2 = axes[1].imshow(noisy_image)
    axes[1].set_title('Image with Noise Applied')
    axes[1].axis('off')

    normalized_noise = (noise_to_plot - np.min(noise_to_plot)) / (np.max(noise_to_plot) - np.min(noise_to_plot))
    im3 = axes[2].imshow(normalized_noise)
    axes[2].set_title('Normalized Noise')
    axes[2].axis('off')

    plt.show()
