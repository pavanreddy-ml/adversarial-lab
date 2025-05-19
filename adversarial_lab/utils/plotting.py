import io
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adversarial_lab.db import DB

from typing import Any, List, Literal, Optional, Dict


class Plotting:
    @staticmethod
    def plot_images_and_noise(image: np.ndarray,
                              noise: np.ndarray,
                              include: Optional[List[Literal[
                                  "image", "noise", "normalized_noise", "noised_image"
                              ]]] = None,
                              config: Optional[dict] = None) -> None:
        config = config or {}
        include = include or ["image", "noise", "normalized_noise", "noised_image"]

        if image.shape[0] == 1:
            image = image[0]
        if noise.shape[0] == 1:
            noise = noise[0]

        def normalize_for_display(img):
            if np.issubdtype(img.dtype, np.integer):
                return img / 255.0
            elif np.issubdtype(img.dtype, np.floating):
                return (img - img.min()) / (img.max() - img.min() + 1e-8)
            return img

        plots = []
        titles = []

        if "image" in include:
            image_disp = normalize_for_display(image)
            plots.append(image_disp)
            titles.append("Image")

        if "noised_image" in include:
            noisy_image = image + noise
            noisy_image_disp = normalize_for_display(noisy_image)
            plots.append(noisy_image_disp)
            titles.append("Image with Noise Applied")

        if "noise" in include:
            base = np.full_like(noise, 127.0)
            noise_display = base + noise
            plots.append(noise_display.astype("uint8"))
            titles.append("Noise")

        if "normalized_noise" in include:
            norm_noise = (noise - np.min(noise)) / (np.ptp(noise) + 1e-8)
            plots.append(norm_noise)
            titles.append("Normalized Noise")

        n = len(plots)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, img, title in zip(axes, plots, titles):
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            if config.get("show_subtitle", True):
                ax.set_title(title)
            ax.axis('off')

        if config.get("title") is not None:
            fig.suptitle(config["title"])

        plt.tight_layout()
        plt.show()


class PlottingFromDB:
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

    def _get_column(self, column: str) -> pd.DataFrame:
        query = f"SELECT epoch_num, {column} FROM {self.table_name};"
        try:
            rows = self.db.execute_query(query)
            if rows is None:
                raise ValueError(
                    f"No data returned for column '{column}' from the database.")
            df = pd.DataFrame(rows)
        except Exception as e:
            raise ValueError(
                f"Column '{column}' does not exist in the database.") from e

        try:
            df[column] = df[column].apply(json.loads)
        except Exception:
            pass
        return df

    def plot_losses(self,
                    plot_total_loss: bool = True,
                    plot_main_loss: bool = True,
                    plot_penalties: bool = True,
                    penalty_idx: Optional[List[int]] = None,
                    config: Optional[dict] = None):
        config = config or {}

        df = self._get_column("epoch_losses")
        fig = go.Figure()
        keys_seen = {}

        for _, row in df.iterrows():
            epoch = row["epoch_num"]

            if not isinstance(row["epoch_losses"], dict):
                continue

            for key, val in row["epoch_losses"].items():
                if not (
                    (plot_total_loss and key == "Total Loss") or
                    (plot_main_loss and key.startswith("Loss ")) or
                    (plot_penalties and key.startswith("Penalty "))
                ):
                    continue

                if penalty_idx is not None and key.startswith("Penalty "):
                    try:
                        idx = int(key.split("Penalty ")[1].split(":")[0])
                        if idx not in penalty_idx:
                            continue
                    except (ValueError, IndexError):
                        continue

                if key not in keys_seen:
                    keys_seen[key] = {"x": [], "y": []}
                keys_seen[key]["x"].append(epoch)
                keys_seen[key]["y"].append(val)

        if penalty_idx is not None and not any(k.startswith(f"Penalty {penalty_idx}:") for k in keys_seen):
            warnings.warn(
                f"Penalty index {penalty_idx} not found in any epoch.")

        for key, vals in keys_seen.items():
            fig.add_trace(go.Scatter(
                x=vals["x"], y=vals["y"], mode="lines+markers", name=key))

        fig.update_layout(
            title=config.get(
                "title", "Losses Over Epochs") if self.plot_config["show_title"] else None,
            xaxis_title=config.get("xaxis_title", "Epoch"),
            yaxis_title=config.get("yaxis_title", "Loss Value"),
            template=self.plot_config["template"],
            height=self.plot_config["height"],
            width=self.plot_config["width"],
            showlegend=self.plot_config["show_legend"]
        )
        fig.show()

    def plot_noise_statistics(self,
                          stats: Optional[List[Literal[
                              "mean", "median", "std", "min", "max", "var",
                              "p25", "p75", "p_custom_x", "iqr", "skew", "kurtosis"
                          ]]] = None,
                          config: Optional[dict] = None):
        config = config or {}
        all_stats = [
            "mean", "median", "std", "min", "max", "var", "p25", "p75",
            "p_custom_x", "iqr", "skew", "kurtosis"
        ]
        stats = stats or all_stats

        df = self._get_column("epoch_noise_stats")
        if df.empty:
            return

        keys_seen = {}
        for _, row in df.iterrows():
            epoch = row["epoch_num"]
            stats_dict = row["epoch_noise_stats"]
            if not isinstance(stats_dict, dict):
                continue

            for stat in stats:
                if stat not in stats_dict:
                    continue
                if stat not in keys_seen:
                    keys_seen[stat] = {"x": [], "y": []}
                keys_seen[stat]["x"].append(epoch)
                keys_seen[stat]["y"].append(stats_dict[stat])

        fig = make_subplots(
            rows=len(keys_seen), cols=1,
            shared_xaxes=True,
            subplot_titles=[f"{stat} Over Epochs" for stat in keys_seen]
        )

        for i, (stat, vals) in enumerate(keys_seen.items(), start=1):
            fig.add_trace(go.Scatter(
                x=vals["x"], y=vals["y"], mode="lines+markers", name=stat),
                row=i, col=1
            )
            fig.update_yaxes(title_text="Value", row=i, col=1)

        fig.update_layout(
            title=config.get("title", "Noise Statistics Over Epochs") if self.plot_config["show_title"] else None,
            xaxis_title=config.get("xaxis_title", "Epoch"),
            template=self.plot_config["template"],
            height=self.plot_config["height"] * len(keys_seen),
            width=self.plot_config["width"],
            showlegend=self.plot_config["show_legend"]
        )
        fig.show()

    def plot_predictions(self, 
                     n: int = 10, 
                     idx: Optional[List[int]] = None,
                     config: Optional[dict] = None
                     ) -> None:
        if n > 20:
            warnings.warn("Requested top-k predictions exceeds maximum of 20. Defaulting to 20.")
            n = 20

        if idx is not None:
            if len(idx) > 20:
                warnings.warn("Provided index list exceeds maximum of 20. Using first 20 entries.")
                idx = idx[:20]
            idx = [str(i) for i in idx]

        config = config or {}
        df = self._get_column("epoch_predictions")
        fig = go.Figure()

        keys_seen: Dict[str, Dict[str, List]] = {}
        for _, row in df.iterrows():
            epoch = row["epoch_num"]
            predictions: Dict[str, float] = row["epoch_predictions"]
            if not isinstance(predictions, dict):
                continue

            if idx is not None:
                selected = [(k, predictions[k]) for k in idx if k in predictions]
            else:
                selected = sorted(predictions.items(), key=lambda x: -x[1])[:n]

            for pred_idx, val in selected:
                if pred_idx not in keys_seen:
                    keys_seen[pred_idx] = {"x": [], "y": []}
                keys_seen[pred_idx]["x"].append(epoch)
                keys_seen[pred_idx]["y"].append(val)

        for pred_idx in sorted(keys_seen, key=lambda x: int(x)):
            vals = keys_seen[pred_idx]
            fig.add_trace(go.Scatter(
                x=vals["x"], y=vals["y"], mode="lines+markers", name=f"Idx {pred_idx}"))

        fig.update_layout(
            title=config.get("title", "Top Predictions Over Epochs") if self.plot_config["show_title"] else None,
            xaxis_title=config.get("xaxis_title", "Epoch"),
            yaxis_title=config.get("yaxis_title", "Prediction Value"),
            template=self.plot_config["template"],
            height=self.plot_config["height"],
            width=self.plot_config["width"],
            showlegend=self.plot_config["show_legend"]
        )
        fig.show()

    def plot_sample_and_noise(self,
                          include: Optional[List[Literal[
                              "original_image", "preprocessed_image",
                              "noise", "normalized_noise", "noised_sample"
                          ]]] = None,
                          epoch: Optional[int] = None):
        include = include or ["original_image", "preprocessed_image", "noise", "normalized_noise", "noised_sample"]
        computed_columns = {"normalized_noise", "noised_sample"}
        required_columns = (set(include) - computed_columns) | {"noise"}

        epochs = self.db.execute_query(f"SELECT epoch_num FROM {self.table_name} ORDER BY epoch_num ASC")
        if not epochs:
            raise RuntimeError("No epoch data found.")

        all_epochs = [e["epoch_num"] for e in epochs]
        epoch = epoch if epoch is not None else (all_epochs[-2] if len(all_epochs) >= 2 else all_epochs[0])
        target_epoch = (0, epoch)

        cols = ['epoch_num'] + list(required_columns)
        placeholders = ','.join(['?'] * len(target_epoch))
        query = f"SELECT {', '.join(cols)} FROM {self.table_name} WHERE epoch_num IN ({placeholders})"
        row_data = self.db.execute_query(query, target_epoch)
        if not row_data:
            raise ValueError(f"No data found for epoch {target_epoch}")
        zero_row = [row for row in row_data if row["epoch_num"] == 0][0]
        epoch_row = [row for row in row_data if row["epoch_num"] == epoch][0]

        blobs = {}
        for col in required_columns:
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
        plt.show()