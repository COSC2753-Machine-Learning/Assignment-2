import math
import os
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt

def create_countplot(
        df: pd.DataFrame,
        col: str,
        ax,
        horizontal: bool = False,
        title: str = None,
        annotate: bool = False,
        palette=None,
        xticklabels_rotation: float = 0.0,
) -> None:
    if horizontal:
        sns.countplot(y=df[col], ax=ax, palette=palette)
    else:
        sns.countplot(x=df[col], ax=ax, palette=palette)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xticklabels_rotation)

    if title is not None:
        ax.set_title(title)

    for patch in ax.patches:
        if annotate:
            if horizontal:
                ax.annotate(
                    str(math.floor(patch.get_width())),
                    (
                        patch.get_width() / 2.,
                        patch.get_y() + patch.get_height() / 2.
                    ),
                    ha='center', va='center', xytext=(0, 0), textcoords='offset points'
                )
            else:
                ax.annotate(
                    str(math.floor(patch.get_height())),
                    (
                        patch.get_x() + patch.get_width() / 2.,
                        patch.get_height()
                    ),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points'
                )


def create_histogram(
        df: pd.DataFrame,
        col: str,
        ax,
        title: str = None,
        stat: str = 'count',
        bins='auto',
        kde: bool = False,
        line_kws: dict = None,
        annotate: bool = False,
        palette=None,
        xticklabels_rotation: float = 0.0,
) -> None:
    sns.histplot(df, x=col, ax=ax, bins=bins, kde=kde, line_kws=line_kws, stat=stat, palette=palette)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xticklabels_rotation)
    if title is not None:
        ax.set_title(title)

    for patch in ax.patches:
        if annotate:
            ax.annotate(
                str(patch.get_height()),
                (
                    patch.get_x() + patch.get_width() / 2.,
                    patch.get_height()
                ),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points'
            )


def create_k_samples(from_dir: str, df: pd.DataFrame, to_file: str = None, k: int = 16):
    # Normalize directory path
    from_dir = os.path.normpath(from_dir) + os.sep

    # Calculate grid dimensions
    cols = int(math.sqrt(k))
    rows = int(math.ceil(k / cols))

    # Ensure even distribution of indices even if df has fewer rows than k
    sample_indices = np.linspace(0, df.shape[0] - 1, min(k, df.shape[0]), dtype=int)

    # Create subplots
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])
    ax = ax.ravel()  # Flatten the axis array for easy iteration

    # Load and display images
    for ax_i, idx in enumerate(sample_indices):
        img_path = os.path.join(from_dir, df.iloc[int(idx)]['ImgPath'])
        try:
            with Image.open(img_path) as im:
                # Optional: Resize image to fit the subplot
                im = im.resize((int(fig.bbox_inches.width * fig.dpi / cols),
                                int(fig.bbox_inches.height * fig.dpi / rows)), Image.LANCZOS)

                ax[ax_i].imshow(im)
                ax[ax_i].set_title(os.path.basename(img_path), fontsize=10)
                ax[ax_i].axis('off')
        except IOError:
            ax[ax_i].text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=10)
            ax[ax_i].set_title(os.path.basename(img_path), fontsize=8)
            ax[ax_i].axis('off')

    # Hide any unused axes if k is less than rows*cols
    for j in range(ax_i + 1, len(ax)):
        ax[j].axis('off')

    plt.tight_layout()
    plt.show()

    # Save the figure to a file if specified
    if to_file:
        fig.savefig(to_file)


def create_learning_curve(
        train_loss, val_loss, train_metric, val_metric,
        to_file: str = None
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(train_loss, 'r--')
    ax[0].plot(val_loss, 'b--')
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend(['train', 'val'])

    ax[1].plot(train_metric, 'r--')
    ax[1].plot(val_metric, 'b--')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].axhline(y=0.125, c='g', alpha=0.5)  # Random probability - naive classifier
    ax[1].legend(['train', 'val', 'random baseline'])

    fig.tight_layout()
    plt.show()
    if to_file is not None:
        fig.savefig(to_file)
