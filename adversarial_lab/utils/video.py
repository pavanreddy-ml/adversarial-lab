import io
import pickle
import imageio
import warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from adversarial_lab.db import DB

from typing import List, Optional, Literal, Tuple

class VideoPlotter:
    def __init__(self, db: DB, table_name: str, plot_config: Optional[dict] = None):
        self.db = db
        self.table_name = table_name
        plot_config = plot_config or {}

        self.plot_config = {
            "height": plot_config.get("height", 500),
            "width": plot_config.get("width", 800),
            "template": plot_config.get("template", "plotly_dark"),
            "show_title": plot_config.get("show_title", True),
            "show_legend": plot_config.get("show_legend", True),
        }

    def make_video(self,
               include: Optional[List[Literal["original_image", "preprocessed_image", "noise", "normalized_noise", "noised_sample"]]] = None,
               fps: int = 1,
               save_path: str = "output_video.mp4"):
        include = include or ["original_image", "preprocessed_image", "noise", "normalized_noise", "noised_sample"]
        computed_cols = {"normalized_noise", "noised_sample"}
        required_cols = (set(include) - computed_cols) | {"noise"}

        epochs = self.db.execute_query(f"SELECT epoch_num FROM {self.table_name} ORDER BY epoch_num ASC")
        if not epochs:
            raise RuntimeError("No epoch data found.")

        all_epochs = [e["epoch_num"] for e in epochs]
        if len(all_epochs) < 2:
            raise RuntimeError("At least two epochs required (including epoch 0).")

        target_epochs = tuple([0] + all_epochs[1:-1])
        cols = ["epoch_num"] + list(required_cols)
        placeholders = ",".join(["?"] * len(target_epochs))
        query = f"SELECT {', '.join(cols)} FROM {self.table_name} WHERE epoch_num IN ({placeholders})"
        rows = self.db.execute_query(query, target_epochs)
        if not rows:
            raise RuntimeError("No data found for selected epochs.")

        row_map = {row["epoch_num"]: row for row in rows}
        zero_row = row_map[0]

        frames = []

        for epoch in all_epochs[1:-1]:
            epoch_row = row_map.get(epoch)
            if epoch_row is None:
                continue

            blobs = {}
            for col in required_cols:
                if col in {"original_image", "preprocessed_image"}:
                    blobs[col] = zero_row.get(col)
                elif col in {"noise"}:
                    blobs[col] = epoch_row.get(col)

            decoded = {}
            for col, blob in blobs.items():
                if blob is None:
                    decoded[col] = None
                    continue
                try:
                    if col in ["original_image"]:
                        decoded[col] = np.array(Image.open(io.BytesIO(blob)).convert("RGB"))
                    elif col in ["preprocessed_image", "noise"]:
                        decoded[col] = pickle.loads(blob)
                except Exception as e:
                    warnings.warn(f"Failed to decode {col}: {e}")
                    decoded[col] = None

            images = []
            for col in include:
                noise = decoded.get("noise")
                orig = decoded.get("original_image")
                prep = decoded.get("preprocessed_image")
                try:
                    if col == "preprocessed_image":
                        if prep is not None:
                            arr = prep.astype("float32")
                            min_val, max_val = arr.min(), arr.max()
                            if max_val - min_val < 1e-8:
                                arr[:] = 0
                            else:
                                arr = 2 * (arr - min_val) / (max_val - min_val) - 1
                            arr = ((arr + 1) / 2) * 255
                            arr = arr.astype("uint8")
                            img = Image.fromarray(arr)
                        else:
                            img = None
                    elif col == "noise":
                        arr = decoded["noise"]
                        if arr is not None:
                            arr = arr.astype("float32")
                            base = np.full_like(arr, 127.0)
                            arr = base + arr
                            arr = np.clip(arr, 0, 255)
                            img = Image.fromarray(arr.astype("uint8"))
                        else:
                            img = None
                    elif col == "normalized_noise":
                        arr = decoded["noise"]
                        if arr is not None:
                            arr = (arr - arr.min()) / (arr.ptp() + 1e-8)
                            img = Image.fromarray((arr * 255).astype("uint8"))
                        else:
                            img = None
                    elif col == "noised_sample":
                        base = None
                        if noise is not None:
                            if orig is not None and orig.shape == noise.shape:
                                base = orig
                            elif prep is not None and prep.shape == noise.shape:
                                base = prep
                        if base is not None:
                            arr = np.clip(base.astype("float32") + noise.astype("float32"), 0, 255).astype("uint8")
                            img = Image.fromarray(arr)
                        else:
                            img = None
                    else:
                        arr = decoded.get(col)
                        if arr is None:
                            img = None
                        elif arr.ndim == 2:
                            img = Image.fromarray(arr)
                        elif arr.ndim == 3 and arr.shape[-1] in [1, 3]:
                            img = Image.fromarray(arr.astype("uint8"))
                        else:
                            raise ValueError(f"Unsupported shape for image array: {arr.shape}")
                except Exception as e:
                    warnings.warn(f"Failed to process {col} at epoch {epoch}: {e}")
                    img = None
                images.append(img)

            background_color = "black" if self.plot_config.get("template", "") == "plotly_dark" else "white"
            title_color = "white" if background_color == "black" else "black"

            n_images = len(images)
            cols = 2
            rows = (n_images + cols - 1) // cols
            aspect_ratio = (4, 4)

            fig, axes = plt.subplots(rows, cols,
                                    figsize=(cols * aspect_ratio[0], rows * aspect_ratio[1]),
                                    facecolor=background_color)
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

            for ax, img, title in zip(axes, images, include):
                ax.axis("off")
                if img is not None:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"Epoch {epoch}: {title}", color=title_color)

            for ax in axes[len(images):]:
                ax.axis("off")

            plt.tight_layout(pad=0.1)

            fig.canvas.draw()
            frame = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            frames.append(np.array(frame))
            plt.close(fig)

        if not frames:
            raise RuntimeError("No frames were generated; check your data or 'include' list.")

        writer = imageio.get_writer(save_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        print(f"Video saved to {save_path}")