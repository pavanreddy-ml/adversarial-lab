from typing import Any

import numpy as np
import matplotlib.pyplot as plt

class Plotting:
    @staticmethod
    def plot_images_and_noise(image: np.ndarray, 
                              noise: np.ndarray):
        if image.shape[0] == 1:
            image = image[0]

        if noise.shape[0] == 1:
            noise = noise[0]

        fig, axes = plt.subplots(1, 3, figsize=(20, 10))

        def normalize_for_display(img):
            if np.issubdtype(img.dtype, np.integer):
                return img / 255.0
            elif np.issubdtype(img.dtype, np.floating):
                return (img - img.min()) / (img.max() - img.min() + 1e-8)
            return img

        image_disp = normalize_for_display(image)
        noisy_image = image + noise
        noisy_image_disp = normalize_for_display(noisy_image)

        im1 = axes[0].imshow(image_disp, cmap='gray' if image_disp.ndim == 2 else None)
        axes[0].set_title('Image')
        axes[0].axis('off')

        im2 = axes[1].imshow(noisy_image_disp, cmap='gray' if noisy_image_disp.ndim == 2 else None)
        axes[1].set_title('Image with Noise Applied')
        axes[1].axis('off')

        normalized_noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise) + 1e-8)
        im3 = axes[2].imshow(normalized_noise, cmap='gray' if normalized_noise.ndim == 2 else None)
        axes[2].set_title('Normalized Noise')
        axes[2].axis('off')

        plt.show()
