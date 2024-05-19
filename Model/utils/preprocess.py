import os
import random
import shutil
import imagehash
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, UnidentifiedImageError
import numpy as np

def find_duplicates(df, base_dir, hash_size=8) -> pd.DataFrame:
    image_hashes = {}
    duplicates_count = 0
    base_dir = os.path.normpath(base_dir) + os.sep

    for index, row in df.iterrows():
        img_path = os.path.join(base_dir, row["ImgPath"])
        try:
            with Image.open(img_path) as img:
                # Convert image hash to a hex string immediately
                img_hash = str(imagehash.phash(img, hash_size))
                if img_hash in image_hashes:
                    image_hashes[img_hash].append(row["ImgPath"])
                    duplicates_count += 1
                else:
                    image_hashes[img_hash] = [row["ImgPath"]]
        except IOError as e:
            print(f"Error opening image {img_path}: {e}")

    # Identifying groups with more than one image
    duplicated_image_hashes = {hash_val: paths for hash_val, paths in image_hashes.items() if len(paths) > 1}
    duplicates_list = [(hash_val, path) for hash_val, paths in duplicated_image_hashes.items() for path in paths]
    duplicates_df = pd.DataFrame(duplicates_list, columns=['Hash', 'ImgPath'])

    return duplicates_df

def visualize_duplicates(duplicate_dataframe, num_samples) -> None:
    # Convert the DataFrame to a dictionary
    duplicates_image_hashes = duplicate_dataframe.groupby('Hash')['ImgPath'].apply(list).to_dict()

    # Select random k samples from the dictionary
    sample_indices = random.sample(list(duplicates_image_hashes.keys()), num_samples)

    fig, ax = plt.subplots(num_samples, 2, figsize=(16, 12))

    for iteration, sample_index in enumerate(sample_indices):
        paths = duplicates_image_hashes[sample_index]

        for i, path in enumerate(paths):
            if i >= 2:
                break

            im = Image.open(f'../Data/Furniture_Data/{path}')
            ax[iteration, i].imshow(im)
            if i > 0:
                ax[iteration, i].text(
                    0.5, 0.5, f'Duplication',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax[iteration, i].transAxes,
                    fontsize=14,
                    color='red',
                    weight='bold'
                )
            ax[iteration, i].set_title(path, fontsize=8)
            ax[iteration, i].axis('off')
    
    plt.subplots_adjust(wspace=-0.5, hspace=0.2)
    plt.show()
    

# Data Augmentation

def augment_image(image_path):
    augmentations = [
        flip_image,
        color_jitter,
        add_noise,
        blur_image,
        rotate_image
    ]
    # Open the image using the provided path
    with Image.open(image_path) as image:
        # Randomly choose a subset of augmentations to apply
        num_augmentations = random.randint(1, len(augmentations))
        random_augmentations = random.sample(augmentations, num_augmentations)

        for aug in random_augmentations:
            image = aug(image)

    return image

def rotate_image(image: Image):
    # Randomly select a rotation angle from 90, 180, or 270 degrees
    rotate_angle = random.choice([90, 180, 270])
    
    # Rotate the image with expansion to preserve resolution
    rotated_image = image.rotate(rotate_angle, expand=True)
    return rotated_image

def flip_image(image):
    return ImageOps.mirror(image)

def color_jitter(image):
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    return image

def add_noise(image):
    np_image = np.array(image)
    noise = np.random.normal(0, 10, np_image.shape)
    np_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_image)

def blur_image(image):
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))


# Data Copying
def copy_dataset_from_df(df: pd.DataFrame, source_dir: str, destination_dir: str) -> None:
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    for index, row in df.iterrows():
        src_img_path = os.path.join(source_dir, row['ImgPath'])
        dest_img_path = os.path.join(destination_dir, row['ImgPath'])

        try:
            # Create necessary directories in the destination path
            os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
            # Copy the image file
            shutil.copy2(src_img_path, dest_img_path)
            print(f"Copied {src_img_path} to {dest_img_path}")
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error copying {src_img_path}: {e}")


def augment_duplicated_images(duplicated_df: pd.DataFrame, source_dir: str, augmented_dir: str) -> None:
    os.makedirs(augmented_dir, exist_ok=True)

    for index, row in duplicated_df.iterrows():
        src_img_path = os.path.join(source_dir, row['ImgPath'])
        dest_img_path = os.path.join(augmented_dir, row['ImgPath'])

        try:
            os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
            augmented_image = augment_image(src_img_path)
            augmented_image.save(dest_img_path)
            print(f"Augmented image saved to {dest_img_path}")
        except (UnidentifiedImageError, PermissionError) as e:
            print(f"Error processing {src_img_path}: {e}")
            
def augment_duplicated_images_loop(df: pd.DataFrame, source_dir: str, iter: int) -> None:
    for _ in range(iter):
        duplicates_df = find_duplicates(df, source_dir)
        if duplicates_df.empty:
            print("No more duplicates found!")
            break
        augment_duplicated_images(duplicates_df, source_dir, source_dir)
